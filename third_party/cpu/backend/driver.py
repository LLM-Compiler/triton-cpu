import os
import hashlib
import importlib
import importlib.resources
import tempfile
import time

import triton
import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from pathlib import Path
from triton._C.libtriton import llvm

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

include_dirs = []
library_dirs = [_triton_C_dir]
libraries = ["stdc++"]

# Skip non-existent paths
sys_include_dir = os.path.join(_dirname, "include")
if os.path.exists(sys_include_dir):
    include_dirs.append(sys_include_dir)

sys_lib_dir = os.path.join(_dirname, "lib")
if os.path.exists(sys_lib_dir):
    library_dirs.append(sys_lib_dir)


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    args_format_full = ''.join([format_of(ty) for ty in signature.values()])
    args_format_without_constexpr = ''.join([format_of(ty) for ty in signature.values() if ty != "constexpr"])
    format_full = "iiiOKOOOO" + args_format_full
    format_without_constexpr = "iiiOKOOOO" + args_format_without_constexpr

    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}

    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")

    arg_ptrs_list_full = ', '.join(f"&arg{i}" for i in signature.keys())
    arg_ptrs_list_without_constexpr = ', '.join(f"&arg{i}" for i, ty in signature.items() if ty != "constexpr")
    kernel_fn_args = [i for i, ty in signature.items() if i not in constants and ty != "constexpr"]
    signature_without_constexprs = {i: ty for i, ty in signature.items() if ty != "constexpr"}
    ptr_arg_indices = [i for i, ty in signature_without_constexprs.items() if ty[0] == "*"]
    ptr_arg_pos = {idx: pos for pos, idx in enumerate(ptr_arg_indices)}
    ptr_arg_count = len(ptr_arg_indices)
    num_signature_full = len(signature)
    num_signature_without_constexpr = len(signature_without_constexprs)
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args)
    kernel_fn_arg_types = ', '.join([f"{ty_to_cpp(signature[i])}" for i in kernel_fn_args] + ["uint32_t"] * 6)
    kernel_launch_args_default = ', '.join(
        f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"arg{i}"
        for i, ty in signature_without_constexprs.items()
    )
    kernel_launch_args_numa = ', '.join(
        f"numa_ptrs[{ptr_arg_pos[i]}]" if ty[0] == "*" else f"arg{i}"
        for i, ty in signature_without_constexprs.items()
    )

    # generate glue code
    src = f"""
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <optional>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>
#include <memory>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif // __linux__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

inline bool getBoolEnv(const std::string &env) {{
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) {{ return std::tolower(c); }});
  return str == "on" || str == "true" || str == "1";
}}

inline std::optional<int64_t> getIntEnv(const std::string &env) {{
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;

  char *endptr;
  long int result = std::strtol(cstr, &endptr, 10);
  if (endptr == cstr)
    assert(false && "invalid integer");
  return result;
}}

using kernel_ptr_t = void(*)({kernel_fn_arg_types});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  // TODO: which order would be more effective for cache locality?
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        grids[z * gridY * gridX + y * gridX + x][0] = x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }}
    }}
  }}
  return grids;
}}

static std::vector<int> parse_cpu_list(const std::string& cpu_list) {{
  std::vector<int> cpus;
  std::stringstream ss(cpu_list);
  std::string token;
  while (std::getline(ss, token, ',')) {{
    token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) {{
      return std::isspace(c);
    }}), token.end());
    if (token.empty()) {{
      continue;
    }}
    size_t dash = token.find('-');
    if (dash == std::string::npos) {{
      cpus.push_back(std::stoi(token));
      continue;
    }}
    int start = std::stoi(token.substr(0, dash));
    int end = std::stoi(token.substr(dash + 1));
    if (end < start) {{
      std::swap(start, end);
    }}
    for (int cpu = start; cpu <= end; ++cpu) {{
      cpus.push_back(cpu);
    }}
  }}
  return cpus;
}}

static std::vector<int> get_numa_node_cpus(int node_id) {{
  std::vector<int> cpus;
  std::ifstream file("/sys/devices/system/node/node" + std::to_string(node_id) + "/cpulist");
  if (!file.is_open()) {{
    return cpus;
  }}
  std::string cpu_list;
  std::getline(file, cpu_list);
  if (cpu_list.empty()) {{
    return cpus;
  }}
  try {{
    cpus = parse_cpu_list(cpu_list);
  }} catch (...) {{
    cpus.clear();
  }}
  return cpus;
}}

static void bind_current_thread_to_cpu(int cpu_id) {{
#ifdef __linux__
  if (cpu_id < 0) {{
    return;
  }}
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
  (void)cpu_id;
#endif // __linux__
}}

static void run_omp_kernels(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, kernel_ptr_t kernel_ptr {(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  // TODO: Consider using omp collapse(3) clause for simplicity?
  size_t N = gridX * gridY * gridZ;
  if (N == 1) {{
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} 0, 0, 0, 1, 1, 1);
      return;
  }}

  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  // Don't pay OMP overhead price when a single thread is used.
  if (max_threads == 1) {{
    for (size_t i = 0; i < N; ++i) {{
      const auto [x, y, z] = all_grids[i];
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }}
    return;
  }}

  // For now, use the default chunk size, total iterations / max_threads.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(max_threads)
#endif // _OPENMP
  for (size_t i = 0; i < N; ++i) {{
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
  const Py_ssize_t expected_with_constexpr = 9 + {num_signature_full};
  const Py_ssize_t expected_without_constexpr = 9 + {num_signature_without_constexpr};
  if (expected_with_constexpr == expected_without_constexpr) {{
    if(!PyArg_ParseTuple(args, \"{format_full}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                         &kernel_metadata, &launch_metadata,
                                         &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list_full if len(signature) > 0 else ''})) {{
      return NULL;
    }}
  }} else if (nargs == expected_without_constexpr) {{
    if(!PyArg_ParseTuple(args, \"{format_without_constexpr}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                         &kernel_metadata, &launch_metadata,
                                         &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list_without_constexpr if len(signature_without_constexprs) > 0 else ''})) {{
      return NULL;
    }}
  }} else if (nargs == expected_with_constexpr) {{
    if(!PyArg_ParseTuple(args, \"{format_full}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                         &kernel_metadata, &launch_metadata,
                                         &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list_full if len(signature) > 0 else ''})) {{
      return NULL;
    }}
  }} else {{
    PyErr_Format(
        PyExc_TypeError,
        "launch: expected %zd (without constexpr) or %zd (with constexpr) arguments, got %zd",
        expected_without_constexpr,
        expected_with_constexpr,
        nargs);
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  // Extract num_threads metadata.
  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);
  Py_XDECREF(num_threads_attr);

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};
  bool launched_numa = false;
  if ({ptr_arg_count} > 0 && PyObject_HasAttrString(kernel_metadata, "numa_args")) {{
    PyObject* numa_args_attr = PyObject_GetAttrString(kernel_metadata, "numa_args");
    if (numa_args_attr != NULL && numa_args_attr != Py_None && PySequence_Check(numa_args_attr)) {{
      Py_ssize_t numa_count = PySequence_Size(numa_args_attr);
      if (numa_count > 0) {{
        std::vector<int> numa_nodes((size_t)numa_count);
        for (Py_ssize_t i = 0; i < numa_count; ++i) {{
          numa_nodes[(size_t)i] = static_cast<int>(i);
        }}
        for (const char* node_attr_name : {{"numa_list", "numa_nodes", "numa_node_list"}}) {{
          if (!PyObject_HasAttrString(kernel_metadata, node_attr_name)) {{
            continue;
          }}
          PyObject* numa_nodes_attr = PyObject_GetAttrString(kernel_metadata, node_attr_name);
          if (numa_nodes_attr != NULL && PySequence_Check(numa_nodes_attr) &&
              PySequence_Size(numa_nodes_attr) == numa_count) {{
            bool valid_nodes = true;
            for (Py_ssize_t i = 0; i < numa_count; ++i) {{
              PyObject* node = PySequence_GetItem(numa_nodes_attr, i);
              if (node == NULL || !PyLong_Check(node)) {{
                valid_nodes = false;
                Py_XDECREF(node);
                break;
              }}
              numa_nodes[(size_t)i] = (int)PyLong_AsLong(node);
              Py_DECREF(node);
            }}
            Py_DECREF(numa_nodes_attr);
            if (valid_nodes) {{
              break;
            }}
          }} else {{
            Py_XDECREF(numa_nodes_attr);
          }}
        }}

        std::vector<std::vector<void*>> per_node_ptrs(
            (size_t)numa_count, std::vector<void*>((size_t){ptr_arg_count}, nullptr));
        bool valid_numa_args = true;
        for (Py_ssize_t node_i = 0; node_i < numa_count && valid_numa_args; ++node_i) {{
          PyObject* node_args = PySequence_GetItem(numa_args_attr, node_i);
          if (node_args == NULL || !PySequence_Check(node_args) ||
              PySequence_Size(node_args) != {ptr_arg_count}) {{
            PyErr_Format(
                PyExc_TypeError,
                "kernel_metadata.numa_args[%zd] must be a sequence of length {ptr_arg_count}",
                node_i);
            Py_XDECREF(node_args);
            valid_numa_args = false;
            break;
          }}
          for (Py_ssize_t ptr_i = 0; ptr_i < {ptr_arg_count}; ++ptr_i) {{
            PyObject* ptr_obj = PySequence_GetItem(node_args, ptr_i);
            if (ptr_obj == NULL) {{
              valid_numa_args = false;
              break;
            }}
            DevicePtrInfo ptr_info = getPointer(ptr_obj, (int)ptr_i);
            Py_DECREF(ptr_obj);
            if (!ptr_info.valid) {{
              valid_numa_args = false;
              break;
            }}
            per_node_ptrs[(size_t)node_i][(size_t)ptr_i] = ptr_info.dev_ptr;
          }}
          Py_DECREF(node_args);
        }}

        if (valid_numa_args) {{
          const size_t N = (size_t)gridX * (size_t)gridY * (size_t)gridZ;
          auto all_grids = get_all_grids(gridX, gridY, gridZ);

          int hw_threads = (int)std::thread::hardware_concurrency();
          if (hw_threads <= 0) {{
            hw_threads = 1;
          }}
          int total_threads = (num_threads > 0) ? num_threads : hw_threads;
          if (total_threads < (int)numa_count) {{
            total_threads = (int)numa_count;
          }}

          std::vector<std::thread> workers;
          workers.reserve((size_t)total_threads);
          int base_threads = total_threads / (int)numa_count;
          int rem_threads = total_threads % (int)numa_count;
          for (Py_ssize_t node_i = 0; node_i < numa_count; ++node_i) {{
            int node_threads = base_threads + ((node_i < rem_threads) ? 1 : 0);
            if (node_threads <= 0) {{
              continue;
            }}
            std::vector<int> node_cpus = get_numa_node_cpus(numa_nodes[(size_t)node_i]);
            for (int local_tid = 0; local_tid < node_threads; ++local_tid) {{
              workers.emplace_back([&, node_i, local_tid, node_threads, node_cpus]() {{
                if (!node_cpus.empty()) {{
                  bind_current_thread_to_cpu(node_cpus[(size_t)local_tid % node_cpus.size()]);
                }}
                size_t start = (N * (size_t)local_tid) / (size_t)node_threads;
                size_t end = (N * (size_t)(local_tid + 1)) / (size_t)node_threads;
                const std::vector<void*>& numa_ptrs = per_node_ptrs[(size_t)node_i];
                for (size_t i = start; i < end; ++i) {{
                  const auto [x, y, z] = all_grids[i];
                  (*kernel_ptr)({kernel_launch_args_numa + ', ' if len(signature_without_constexprs) > 0 else ''}x, y, z, gridX, gridY, gridZ);
                }}
              }});
            }}
          }}
          for (auto& worker : workers) {{
            worker.join();
          }}
          launched_numa = true;
        }}
      }}
    }}
    Py_XDECREF(numa_args_attr);
  }}
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if (!launched_numa) {{
    run_omp_kernels(gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + kernel_launch_args_default) if len(signature_without_constexprs) > 0 else ''});
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_cpu_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_cpu_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class CPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(src, "__triton_cpu_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu", self.get_current_device())

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        return GPUTarget("cpu", cpu_arch, 0)

    def get_device_interface(self):
        return CPUDeviceInterface()

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        def do_bench_cpu(*args, **kwargs):
            if not 'measure_time_with_hooks' in kwargs:
                kwargs['measure_time_with_hooks'] = True
            return do_bench(*args, **kwargs)

        return do_bench_cpu

    def get_empty_cache_for_benchmark(self):
        import torch

        # A typical LLC size for high-end server CPUs are ~400MB.
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    # TODO maybe CPU should do anything here
    def clear_cache(self, cache):
        cache.zero_()
