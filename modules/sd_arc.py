import collections
from modules import devices
import time
import os
import gc
import torch
from modules.shared import cmd_opts
from modules import shared
from modules import paths
"""
use in three places via a SpecifiedCache model.
1 reload_model_weights
2  

"""
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))


def get_memory():
    try:
        import os
        import psutil
        process = psutil.Process(os.getpid())
        res = process.memory_info() # only rss is cross-platform guaranteed so we dont rely on other values
        ram_total = 100 * res.rss / process.memory_percent() # and total memory is calculated as actual value is not cross-platform safe
        ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
    except Exception as err:
        ram = { 'error': f'{err}' }
    try:
        import torch
        if torch.cuda.is_available():
            s = torch.cuda.mem_get_info()
            system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
            s = dict(torch.cuda.memory_stats(shared.device))
            allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
            reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
            active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
            inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
            warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
            cuda = {
                'system': system,
                'active': active,
                'allocated': allocated,
                'reserved': reserved,
                'inactive': inactive,
                'events': warnings,
            }
        else:
            cuda = {'error': 'unavailable'}
    except Exception as err:
        cuda = {'error': f'{err}'}
    return dict(ram=ram, cuda=cuda)


class SpecifiedCache:
    def __init__(self, checkpoints_loaded) -> None:
        sysinfo = get_memory()
        print(f"system info: {sysinfo}")
        gpu_memory_size = sysinfo.get('cuda',{}).get('system', {}).get('total', 24*1024**3)/1024**3
        ram_size = sysinfo.get('ram',{}).get('total', 32*1024**3)/1024**3
        print(f"gpu memory：{gpu_memory_size : .2f} GB, ram:{ram_size : .2f} GB")

        self.gpu_memory_size = gpu_memory_size
        self.model_size = 5.5 if cmd_opts.no_half else (2.56 if cmd_opts.no_half_vae else 2.39)
        self.size_base = 2.5 if cmd_opts.no_half or cmd_opts.no_half_vae else 0.5
        self.batch_base = 0.3
        self.gpu_lru_size = int((gpu_memory_size - 3) / self.model_size) # 3GB keep.
        self.ram_lru_size = (ram_size - 4.3 * self.gpu_lru_size ) // 8

        self.lru = collections.OrderedDict()
        self.k_lru = self.gpu_lru_size
        rectified_cache = (shared.opts.sd_checkpoint_cache * 5 - 4.3 * self.gpu_lru_size) // 5
        self.k_ram = max(min(self.ram_lru_size, rectified_cache), 0)
        print(f"maximum model in gpu memory：{self.k_lru}，maximum model in ram memory {self.k_ram}")
        self.gpu_specified_models = None
        self.ram_specified_models = None
        self.reload_time = {}
        self.checkpoints_loaded = checkpoints_loaded
    
    def set_specified(self, gpu_filenames: list, ram_filenames: list):
        not_exist = []
        for filename in gpu_filenames + ram_filenames:
            if not os.path.exists(os.path.join(model_path, filename)):
                not_exist.append(filename)
        self.gpu_specified_models = gpu_filenames
        self.ram_specified_models = ram_filenames
        return not_exist

    def is_gpu_specified(self, key):
        return self.gpu_specified_models is None or self.get_model_name(key) in self.gpu_specified_models
    
    def is_ram_specified(self, key):
        return self.ram_specified_models is None or self.get_model_name(key) in self.ram_specified_models


    def lru_get(self, key):
        value =  self.lru.get(key)
        if self.is_cuda(value):
            return value
        return None


    def get(self, key):
        self.reload(key)
        if self.lru.get(key) is not None:
            return self.lru_get(key)   
        return None
    
    def is_cuda(self, value):
        return 'cuda' in str(value.device)

    def contains(self, key):
        return key in self.lru
    
    def delete_oldest(self):
        if len(self.lru) == 0:
            return 
        cudas = [k for k, v in self.lru.items() if self.is_cuda(v)] 
        sorted_cudas = sorted(cudas, key = lambda x: self.reload_time.get(x, 0))
        oldest = sorted_cudas[0]
        del sorted_cudas
        del cudas
        print(f"delete cache: {oldest}")
        v = self.lru.pop(oldest)
        del oldest
        # v.to(devices.cpu)
        del v
        gc.collect()
        devices.torch_gc()
        torch.cuda.empty_cache()

    
    def pop(self, key):
        if key not in self.lru:
            return 
        v = self.lru.pop(key)
        del v
        gc.collect()
        devices.torch_gc()
        torch.cuda.empty_cache()

    def prepare_memory(self):
        if len(self.lru) >= self.k_lru:
            self.delete_oldest()

    def put_lru(self, key, value):
        """
        value must be cuda.
        """
        self.prepare_memory()
        print(f"add cache: {key}")
        self.lru[key] = value    
    
    def get_model_name(self, key):
        return os.path.basename(key)

    def put(self, key, value): 
        if self.contains(key):
            return
        
        if self.is_gpu_specified(key) or len(self.lru) < self.k_lru:
            self.put_lru(key, value)
            return 
        print(f"not cache: {key}")
        del value

    def reload(self, key):
        self.reload_time[key] = time.time_ns()

    def release_memory(self, p):
        gc.collect()
        devices.torch_gc()
        torch.cuda.empty_cache()
        try:
            need_size = (p.height * p.width /(512*512) - 1) * (self.size_base + self.batch_base) + 4 # not include model size
            keep_models_num = int((self.gpu_memory_size - need_size) / self.model_size)
            while len(self.lru) > keep_models_num and len(self.lru) > 0:
                self.delete_oldest()
            print(f"prepare memory: {need_size:.2f} GB")
        except Exception as e:
            raise e

    def put_checkpoint(self, checkpoint_info, state_dict):
        if self.is_ram_specified(checkpoint_info.filename):
            while len(self.checkpoints_loaded) >= self.k_ram:
                self.pop_checkpoint()
            self.checkpoints_loaded[checkpoint_info] = state_dict
            print(f"add checkpoint: {checkpoint_info.filename}")
            return
        del state_dict
        gc.collect()
        
    def pop_checkpoint(self,):
        if len(self.checkpoints_loaded) == 0:
            return
        ckpts = [k for k in self.checkpoints_loaded.keys()] 
        sorted_rams = sorted(ckpts, key = lambda x: self.reload_time.get(x.filename, 0))
        oldest = sorted_rams[0]
        del ckpts
        del sorted_rams
        print(f"delete checkpoint: {oldest.filename}")
        v = self.checkpoints_loaded.pop(oldest)
        del v   
        del oldest  
        gc.collect()

    def get_cudas(self):
        return [self.get_model_name(i) for i in self.lru.keys()] 
    
    def get_rams(self):
        return [self.get_model_name(i.filename) for i in self.checkpoints_loaded.keys()]
