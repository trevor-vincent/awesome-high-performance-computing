<p align="center">
<img src="https://www.montana.edu/uit/rci/assets/hpc.png" width="600">
</p>
<p align="center">
A curated list of awesome high performance computing resources. 
</p>
<p align="center">
  <a href="https://awesome.re">
    <img src="https://awesome.re/badge.svg" />
  </a>
</p>

## Table of Contents

 - [General Info](#general-info)
 - [Software](#software)
 - [Hardware](#hardware) 
 - [People](#people)
 - [Resources](#resources)
 - [Other Curated Lists](#other-curated-lists)
 - [Acknowledgements](#acknowledgements )

## General Info

### A Few Upcoming Supercomputers 
 - [Tianhe-3](https://www.nextplatform.com/2019/05/02/china-fleshes-out-exascale-design-for-tianhe-3/) - 2022, ~700 Petaflop (Linpack500)
 - [Venado](https://discover.lanl.gov/news/0530-venado/) - 2024, Grace-Hopper based ~10 exaflops
   
### Most Recent List of the Top500 Supercomputers
 - [Top500 (Nov. 2024)](https://www.top500.org/lists/top500/2024/11/)
 - [HPCG Top500 (Nov. 2024)](https://www.top500.org/lists/hpcg/2024/11/)
 - [Green500 (Nov. 2024)](https://www.top500.org/lists/green500/2024/11/)
 - [io500](https://io500.org/)
 
### History
 - [History of Supercomputing (Wikipedia)](https://en.wikipedia.org/wiki/History_of_supercomputing)
 - [History of Parallel Computing (Wikipedia)](https://en.wikipedia.org/wiki/Parallel_computing#History)
 - [History of the Top500 (Wikipedia)](https://en.wikipedia.org/wiki/TOP500)
 - [History of LLNL Computing](https://computing.llnl.gov/about/machine-history)
 - [The Supermen: The Story of Seymour Cray ... (1997)](https://www.amazon.ca/Supermen-Seymour-Technical-Wizards-Supercomputer/dp/0471048852/ref=sr_1_1?crid=1IOWC3IOYWPOP&keywords=seymour+cray&qid=1690959561&sprefix=seymour+cray%2Caps%2C88&sr=8-1)
 - [Unmatched - 50 Years of Supercomputing (2023)](https://www.routledge.com/Unmatched-50-Years-of-Supercomputing/Barkai/p/book/9780367479619)
   
### Trends
 - [Trends in HPC for AI workloads](https://epochai.org/trends)
 
## Software

#### Popular HPC Programming Libraries/APIs/Tools/Standards/Simulators
- [alpaka](https://github.com/alpaka-group/alpaka) - The alpaka library is a header-only C++17 abstraction library for accelerator development
- [async-rdma](https://github.com/datenlord/async-rdma) - A framework for writing RDMA applications with high-level abstraction and asynchronous APIs
- [CAF](https://github.com/actor-framework/actor-framework) - An Open Source Implementation of the Actor Model in C++
- [Chapel](https://chapel-lang.org/) - A Programming Language for Productive Parallel Computing on Large-scale Systems
- [Charm++](http://charm.cs.illinois.edu/research/charm) - Parallel Programming with Migratable Objects
- [Cilk Plus](https://www.cilkplus.org/) - C/C++ Extension for Data and Task Parallelism
- [Codon](https://github.com/exaloop/codon) - high-performance Python compiler that compiles Python code to native machine code without any runtime overhead
- [CUDA](https://developer.nvidia.com/cuda-toolkit) - High performance NVIDIA GPU acceleration
- [dask](https://dask.org) - Dask provides advanced parallelism for analytics, enabling performance at scale for the tools you love
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - An easy-to-use deep learning optimization software suite that enables unprecedented scale and speed for Deep Learning Training and Inference
- [DeterminedAI](https://www.determined.ai/) - Distributed deep learning
- [FastFlow](https://github.com/fastflow/fastflow) - High-performance Parallel Patterns in C++
- [Galois](https://github.com/IntelligentSoftwareSystems/Galois) - A C++ Library to Ease Parallel Programming with Irregular Parallelism
- [Halide](https://halide-lang.org/index.html#gettingstarted) - A language for fast, portable computation on images and tensors
- [Heteroflow](https://github.com/Heteroflow/Heteroflow) - Concurrent CPU-GPU Task Programming using Modern C++
- [highway](https://github.com/google/highway) - Performance portable SIMD intrinsics
- [HIP](https://github.com/ROCm-Developer-Tools/HIP) - HIP is a C++ Runtime API and Kernel Language for AMD/Nvidia GPU
- [HPC-X](https://developer.nvidia.com/networking/hpc-x) - Nvidia implementation of MPI
- [HPX](https://github.com/STEllAR-GROUP/hpx) - A C++ Standard Library for Concurrency and Parallelism
- [Horovod](https://github.com/horovod/horovod) - Distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet
- [ISPC](https://ispc.github.io/) - An open-source compiler for high-performance SIMD programming on the CPU and GPU
- [Intel ISPC](https://github.com/ispc/ispc) - SPMD compiler
- [Intel TBB](https://www.threadingbuildingblocks.org/) - Threading Building Blocks
- [joblib](https://joblib.readthedocs.io/en/latest/why.html) - Data-flow programming for performance (python)
- [Kompute](https://github.com/KomputeProject/kompute) - The general purpose GPU compute framework for cross vendor graphics cards (AMD, Qualcomm, NVIDIA & friends)
- [Kokkos](https://github.com/kokkos/kokkos) - A C++ Programming Model for Writing Performance Portable Applications on HPC platforms
- [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) - MPI Operator for Kubeflow
- [Legate](https://github.com/nv-legate/legate.numpy) - Nvidia replacement for numpy based on Legion
- [Legion](https://github.com/StanfordLegion/legion) - Distributed heterogeneous programming library
- [MAGMA](https://developer.nvidia.com/magma) - Next generation linear algebra (LA) GPU accelerated libraries
- [Merlin](https://merlin.readthedocs.io/en/latest/) - A distributed task queuing system, designed to allow complex HPC workflows to scale to large numbers of simulations
- [Metal](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu) - Apple's GPU API
- [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) - Microsoft's implementation of MPI
- [MOGSLib](https://github.com/ECLScheduling/MOGSLib) - User defined schedulers
- [mpi4jax](https://github.com/mpi4jax/mpi4jax) - Zero-copy mpi for jax arrays
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) - Python bindings for MPI
- [MPI](https://www.open-mpi.org/) - OpenMPI implementation of the Message passing interface
- [MPI](https://www.mpich.org/) - MPICH implementation of the Message passing interface
- [MPI Standardization Forum](https://www.mpi-forum.org/) - Forum for MPI standardization
- [MPAVICH](https://mvapich.cse.ohio-state.edu/) - Implementation of MPI
- [NCCL](https://developer.nvidia.com/nccl) - The NVIDIA Collective Communication Library for multi-GPU and multi-node communication
- [cuNumeric](https://developer.nvidia.com/cunumeric) - GPU drop-in for numpy
- [stdpar](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/) - GPU accelerated C++ from NVIDIA
- [numba](https://numba.pydata.org/) - A JIT compiler that translates a subset of Python into fast machine code
- [oneAPI](https://www.oneapi.io/) - A unified, multiarchitecture, multi-vendor programming model
- [OpenACC](https://www.openacc.org/) - "OpenMP for GPUs"
- [OpenCilk](https://www.opencilk.org/) - MIT continuation of Cilk Plus
- [OpenMP](https://www.openmp.org/) - Multi-platform Shared-memory Parallel Programming in C/C++ and Fortran
- [PVM](https://www.csm.ornl.gov/pvm/) - Parallel Virtual Machine: A predecessor to MPI for distributed computing
- [PMIX](https://pmix.github.io/standard) - Standard for process management
- [Pollux](https://github.com/polluxio/pollux-payload) - Message Passing Cloud orchestrator
- [Pyfi](https://github.com/radiantone/pyfi) - Distributed flow and computation system
- [RAJA](https://github.com/LLNL/RAJA) - Architecture and programming model portability for HPC applications
- [RaftLib](https://github.com/RaftLib/RaftLib) - A C++ Library for Enabling Stream and Dataflow Parallel Computation
- [ray](https://www.ray.io/) - Scale AI and Python workloads from reinforcement learning to deep learning
- [ROCM](https://rocmdocs.com/en/latest/) - First open-source software development platform for HPC/Hyperscale-class GPU computing
- [RS MPI](https://rsmpi.github.io/rsmpi/mpi/index.html) - Rust bindings for MPI
- [Scalix](https://github.com/NAGAGroup/Scalix) - Data parallel computing framework
- [Simgrid](https://simgrid.org/) - Simulate cluster/HPC environments
- [SkelCL](https://skelcl.github.io/) - A Skeleton Library for Heterogeneous Systems
- [STAPL](https://parasol.tamu.edu/stapl/) - Standard Template Adaptive Parallel Programming Library in C++
- [STLab](http://stlab.cc/libraries/concurrency/) - High-level Constructs for Implementing Multicore Algorithms with Minimized Contention
- [SYCL](https://www.khronos.org/sycl/) - C++ Abstraction layer for heterogeneous devices
- [Taichi](https://github.com/taichi-dev/taichi) - Parallel programming language for high-performance numerical computations in Python
- [Taskflow](https://github.com/taskflow/taskflow) - A Modern C++ Parallel Task Programming Library
- [The Open Community Runtime](https://wiki.modelado.org/Open_Community_Runtime) - Specification for Asynchronous Many Task systems
- [Transwarp](https://github.com/bloomen/transwarp) - A Header-only C++ Library for Task Concurrency
- [Triton](https://triton-lang.org/main/index.html) - Triton is a language and compiler for parallel programming
- [Tuplex](https://tuplex.cs.brown.edu/) - Blazing fast python data science
- [UCX](https://github.com/openucx/ucx#using-ucx) - Optimized production proven-communication framework
- [Zluda](https://github.com/vosen/ZLUDA) - Run unmodified CUDA applications with near-native performance on Intel AMD GPUs.
- [HyperQueue](https://github.com/It4innovations/hyperqueue) - HyperQueue is a tool designed to simplify execution of large workflows (task graphs) on HPC clusters.
  
#### Cluster Hardware Discovery Tools
- [cpuid](https://en.wikipedia.org/wiki/CPUID) - A software instruction available on Intel, AMD, and other processors that can be used to determine processor type and features.
- [cpuid instruction note](https://www.scss.tcd.ie/~jones/CS4021/processor-identification-cpuid-instruction-note.pdf) - A detailed note on the CPUID instruction used for processor identification.
- [cpufetch](https://github.com/Dr-Noob/cpufetch) - A simple yet fancy CPU architecture fetching tool.
- [gpufetch](https://github.com/Dr-Noob/gpufetch) - A tool similar to cpufetch, but for fetching GPU architecture.
- [intel cpuinfo](https://www.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/command-reference/cpuinfo.html) - Intel tool providing information about the characteristics of Intel CPUs.
- [Likwid](https://github.com/RRZE-HPC/likwid) - Provides all information about the supercomputer/cluster.
- [LIKWID.jl](https://juliaperf.github.io/LIKWID.jl/dev/) - Julia wrapper for LIKWID.
- [openmpi hwloc](https://www.open-mpi.org/projects/hwloc/) - Portable Hardware Locality (hwloc) software project.
- [PRK - Parallel Research Kernels](https://github.com/ParRes/Kernels) - A collection of kernels for parallel programming research.

#### Cluster Management/Tools/Schedulers/Stacks
- [BeeGFS](http://beegfs.io/docs/whitepapers/Introduction_to_BeeGFS_by_ThinkParQ.pdf) - A parallel file system designed for performance-critical environments.
- [Bluebanquise](https://github.com/bluebanquise/bluebanquise) - An open-source cluster management tool.
- [Bright Cluster Manager](https://www.brightcomputing.com/brightclustermanager) - Software for deploying and managing HPC and AI server clusters.
- [Ceph](https://ceph.io/en/) - An open-source distributed storage system.
- [DeepOps](https://github.com/NVIDIA/deepops) - Nvidia's GPU infrastructure and automation tools for Kubernetes and Slurm clusters.
- [E4S - The Extreme Scale HPC Scientific Stack](https://e4s-project.github.io/) - A collection of open-source software packages for HPC environments.
- [Easybuild](https://docs.easybuild.io/en/latest/) - A package manager for HPC/supercomputers.
- [EESSI](https://www.eessi.io) - A shared stack of scientific software installations.
- [Flux framework](https://flux-framework.org/) - A framework for high-performance computing clusters.
- [fpsync](http://www.fpart.org/fpsync/) - A tool for fast parallel data transfer using fpart and rsync.
- [GPFS](https://en.wikipedia.org/wiki/GPFS) - A high-performance parallel file system developed by IBM.
- [Guix](https://hpc.guix.info/) - A package manager for HPC/supercomputers.
- [Intel DAOS](https://daos.io) - A software-defined scale-out object store for HPC applications.
- [LSF](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=lsf-batch-jobs-tasks) - A batch system for HPC and distributed computing environments.
- [Lmod](https://lmod.readthedocs.io/en/latest/) - A Lua-based module system for software environment management on HPC systems.
- [Lustre Parallel File System](https://www.lustre.org/) - A high-performance distributed filesystem for large-scale cluster computing.
- [moosefs](https://moosefs.com/) - A fault-tolerant, highly available, distributed file system.
- [NetApp](www.netapp.com) - Intelligent data infrastructure for various workloads.
- [Open Cluster Scheduler](https://github.com/hpc-gridware/clusterscheduler/) - A scalable HPC/AI workload manager based on SGE.
- [OpenHPC](https://openhpc.community/) - A community-led set of HPC components.
- [OpenOnDemand](https://openondemand.org/) - A web portal for accessing supercomputing resources.
- [OpenPBS](https://www.openpbs.org/) - A software for workload management and job scheduling.
- [OpenXdMod](https://open.xdmod.org/7.5/index.html) - A tool for managing high-performance computing resources.
- [RADIUSS](https://computing.llnl.gov/projects/radiuss) - Rapid Application Development via an Institutional Universal Software Stack.
- [rocks](http://www.rocksclusters.org/) - An open-source Linux cluster distribution.
- [Ruse](https://github.com/JanneM/Ruse) - A tool for managing software environments in HPC clusters.
- [SGE](http://star.mit.edu/cluster/docs/0.93.3/guides/sge.html) - A resource management software for large clusters of computers.
- [Slurm](https://slurm.schedmd.com/overview.html) - A cluster management and job scheduling system for Linux clusters.
- [Spack](https://spack.io/) - A package manager for HPC/supercomputers.
- [sstack](https://gitlab.com/nmsu_hpc/sstack) - A tool to install multiple software stacks such as Spack, EasyBuild, and Conda.
- [Starfish](https://starfishstorage.com/) - Unstructured data management and metadata solution for files and objects.
- [Warewulf](https://warewulf.lbl.gov/) - An operating system provisioning system and cluster management tool.
- [xCat](https://xcat.org/) - A distributed computing management and provisioning tool.
- [XDMoD](https://supremm.xdmod.org/10.0/supremm-overview.html) - An open-source tool for managing high-performance computing resources.
- [Globus Connect](https://www.globus.org/globus-connect) - A fast data transfer tool between supercomputers.
- [Slurm Web](https://slurm-web.com/) - Open source web dashboard for Slurm HPC clusters.
  
#### HPC-specific Operating Systems
- [Kitten](https://www.sandia.gov/app/uploads/sites/210/2022/11/pedretti_lanl11.pdf) - A lightweight kernel designed for high-performance computing. It focuses on providing low noise and predictable performance for HPC applications.
- [McKernel](https://github.com/RIKEN-SysSoft/mckernel) - A hybrid kernel that combines Linux and a lightweight kernel designed to provide high performance for HPC applications.
- [mOS](http://cs.iit.edu/~khale/docs/mos.pdf) - A specialized operating system for high-performance computing, designed to support large-scale, manycore processors.

#### Development/Workflow/Monitoring Tools for HPC

- [Apache Airflow](https://airflow.apache.org/) - A platform to programmatically author, schedule, and monitor workflows.
- [Apptainer (formerly Singularity)](https://singularity.lbl.gov/) - Container platform designed for scientific and high-performance computing (HPC) environments.
- [arbiter2](https://github.com/CHPC-UofU/arbiter2) - Monitors and protects interactive nodes with cgroups.
- [Charliecloud](https://hpc.github.io/charliecloud/) - Lightweight container solution for high-performance computing (HPC).
- [Docker](https://www.docker.com/) - A set of platform as a service products that use OS-level virtualization to deliver software in packages called containers.
- [genv](https://github.com/run-ai/genv) - GPU Environment Management for managing and scheduling GPU resources.
- [Grafana](https://github.com/grafana/grafana) - Open-source platform for monitoring and observability, visualizing metrics.
- [grpc](https://grpc.io/) - A high-performance, open-source universal RPC framework.
- [HPC Rocket](https://github.com/SvenMarcus/hpc-rocket) - Allows submitting Slurm jobs in Continuous Integration (CI) pipelines.
- [HTCondor](https://research.cs.wisc.edu/htcondor/) - An open-source high-throughput computing software framework.
- [Jacamar-ci](https://gitlab.com/ecp-ci/jacamar-ci/-/blob/develop/README.md) - CI/CD tool designed for HPC and scientific computing workflows.
- [Kubernetes](https://kubernetes.io/) - An open-source system for automating deployment, scaling, and management of containerized applications.
- [nextflow](https://www.nextflow.io/) - A workflow framework to deploy data-driven computational pipelines.
- [perun](https://github.com/Helmholtz-AI-Energy/perun) - Energy monitor for HPC systems, focusing on performance and energy efficiency.
- [Prefect](https://www.prefect.io/) - A workflow management system, designed for modern infrastructure and powered by the open-source Prefect Core workflow engine.
- [Prometheus](https://prometheus.io/) - An open-source monitoring system with a dimensional data model, flexible query language, efficient time series database and modern alerting approach.
- [redun](https://github.com/insitro/redun) - Workflow engine that emphasizes simplicity, reliability, and scalability.
- [remora](https://github.com/TACC/remora) - Tool for monitoring and reporting the performance of batch jobs on HPC systems.
- [ruptime](https://github.com/alexmyczko/ruptime) - A utility for monitoring the status of computational jobs and systems.
- [Slurmvision slurm dashboard](https://github.com/Ruunyox/slurmvision) - A dashboard for monitoring and managing Slurm jobs.
- [slurm docker cluster](https://github.com/giovtorres/slurm-docker-cluster) - A Slurm cluster implemented using Docker containers, for development and testing.
- [snakemake](https://snakemake.readthedocs.io/en/stable/) - A workflow management system that reduces the complexity of creating reproducible and scalable data analyses.
- [Stui slurm dashboard for the terminal](https://github.com/mil-ad/stui) - A terminal-based UI for managing and monitoring Slurm clusters.
- [Vaex](https://github.com/vaexio/vaex) - A Python library for lazy Out-of-Core DataFrames (similar to Pandas), to visualize and explore big tabular datasets.

  
#### Debugging Tools for HPC

- [ddt](https://www.arm.com/products/development-tools/server-and-hpc/forge/ddt) - A powerful debugger designed for developers to solve complex problems on multi-threaded and multi-process environments in HPC.
- [marmot MPI checker](https://www.lrz.de/services/software/parallel/marmot/) - A tool for detecting and reporting issues in MPI (Message Passing Interface) applications.
- [python debugging tools](https://wiki.python.org/moin/PythonDebuggingTools) - A collection of tools for debugging Python applications, including pdb and other utilities.
- [seer modern gui for gdb](https://github.com/epasveer/seer) - A graphical user interface for GDB, aiming to improve the debugging experience with modern features and visuals.
- [Summary of C/C++ debugging tools](http://pramodkumbhar.com/2018/06/summary-of-debugging-tools/) - An overview of various debugging tools available for C/C++ applications, focusing on HPC environments.
- [totalview](https://totalview.io/) - A comprehensive source code analysis and debugging tool designed for complex software running on HPC systems, supporting a wide range of languages and architectures.


#### Performance/Benchmark Tools for HPC

- [demonspawn](https://github.com/TACC/demonspawn) - A framework for automated execution of benchmarks and simulations, designed for HPC environments.
- [Google benchmark](https://github.com/google/benchmark) - A microbenchmark support library for C++ that tracks performance over time.
- [HPL benchmark](https://www.netlib.org/benchmark/hpl/) - The High Performance Linpack Benchmark for measuring floating-point computing power of systems.
- [kerncraft](https://github.com/RRZE-HPC/kerncraft) - A tool for analytical modeling of loop performance and cache behavior on HPC systems.
- [NASA parallel benchmark suite](https://www.nas.nasa.gov/software/npb.html) - A set of benchmarks designed to evaluate the performance of parallel supercomputers.
- [papi](https://icl.utk.edu/papi/) - Provides standard APIs for accessing hardware performance counters available on modern microprocessors.
- [scalasca](https://www.scalasca.org/) - A software tool that supports performance analysis of large-scale parallel applications.
- [scalene](https://github.com/plasma-umass/scalene) - A high-performance, high-precision CPU, GPU, and memory profiler for Python.
- [Summary of code performance analysis tools](https://doku.lrz.de/display/PUBLIC/Performance+and+Code+Analysis+Tools+for+HPC) - An overview of tools for analyzing HPC application performance.
- [Summary of profiling tools](https://pramodkumbhar.com/2017/04/summary-of-profiling-tools/) - A comprehensive list of profiling tools for performance analysis in HPC.
- [tau](https://www.cs.uoregon.edu/research/tau/home.php) - TAU (Tuning and Analysis Utilities) is a profiling and tracing toolkit for performance analysis of parallel programs.
- [The Bandwidth Benchmark](https://github.com/RRZE-HPC/TheBandwidthBenchmark/) - A tool for measuring memory bandwidth across various CPUs and systems.
- [vampir](https://vampir.eu/) - A tool for detailed analysis of MPI program executions by visualizing their event traces.
- [bytehound memory profiler](https://github.com/koute/bytehound) - A detailed memory profiler for tracking down memory issues and leaks.
- [Flamegraphs](https://www.brendangregg.com/flamegraphs.html) - Visualization tool for profiling software, allowing quick identification of performance bottlenecks.
- [fio](https://linux.die.net/man/1/fio) - Flexible I/O tester for benchmarking and stress/hardware verification.
- [IBM Spectrum Scale Key Performance Indicators (KPI)](https://github.com/IBM/SpectrumScale_NETWORK_READINESS) - Provides key performance indicators for IBM Spectrum Scale, aiding in performance tuning and monitoring.
- [Ior](https://github.com/hpc/ior) - A parallel file system I/O benchmarking tool used widely in HPC for testing storage systems.
- [ngstress](https://github.com/ColinIanKing/stress-ng) - A versatile tool for stressing various subsystems of a computer to find hardware faults or to benchmark performance.
- [Hotspot](https://github.com/KDAB/hotspot/) - The Linux perf GUI for in-depth performance analysis and visualization of software behavior.
- [mixbench](https://github.com/ekondis/mixbench) - A benchmark suite designed to evaluate CPUs and GPUs across different compute and memory operations.
- [pmu-tools (toplev)](https://github.com/andikleen/pmu-tools) - Performance monitoring tools for modern Intel CPUs, offering detailed insights into hardware and application performance.
- [SPEC CPU Benchmark](https://www.spec.org/benchmarks.html) - A benchmark suite designed to provide a comparative measure of compute-intensive performance across the widest practical range of hardware.
- [STREAM Memory Bandwidth Benchmark](https://www.cs.virginia.edu/stream/) - Measures sustainable memory bandwidth and the corresponding computation rate for simple vector kernels.
- [Intel MPI benchmarks](https://www.intel.com/content/www/us/en/docs/mpi-library/user-guide-benchmarks/2021-2/overview.html) - A set of benchmarks designed to measure the performance and scalability of MPI implementations on Intel architectures.
- [Ohio state MPI benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) - A comprehensive suite of benchmarks for evaluating MPI performance across a variety of message passing patterns and communication protocols.
- [hpctoolkit](http://hpctoolkit.org/man/hpctoolkit.html) - An integrated suite of tools for measurement and analysis of program performance on computers ranging from desktops to supercomputers.
- [core-to-core-latency](https://github.com/nviennot/core-to-core-latency) - A diagnostic tool designed to measure and report the latency between CPU cores, aiding in the optimization of parallel computing tasks.
- [speedscope](https://github.com/jlfwong/speedscope) - An interactive, web-based viewer for performance profiles of software. It supports various formats and provides a flamegraph visualization to identify hot paths efficiently.
- [Differential Flamegraphs](https://www.brendangregg.com/blog/2014-11-09/differential-flame-graphs.html) - A visualization technique developed by Brendan Gregg that highlights differences between performance profiles, making it easier to spot performance regressions or improvements.
- [Hyperfine](https://github.com/sharkdp/hyperfine) - A command-line benchmarking tool that provides a simple and user-friendly means to compare the performance of commands, featuring statistical analysis across multiple runs.
- [Openfoam HPC benchmark](https://develop.openfoam.com/committees/hpc/-/wikis/home) - A benchmarking suite for evaluating the High Performance Computing capabilities of OpenFOAM, an open-source CFD software, under various computational loads.
- [OSU microbenchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) - A collection of microbenchmarks designed to evaluate the performance of MPI implementations across various communication protocols and message sizes.
- [fio flexible I/O tester](https://fio.readthedocs.io/) - A versatile tool for I/O workload simulation and benchmarking, capable of testing a wide array of storage and filesystem configurations.
- [vftrace](https://github.com/SX-Aurora/Vftrace) - A tracing tool specifically designed for the NEC SX-Aurora TSUBASA Vector Engine, enabling detailed performance analysis of vectorized code.
- [tinymembench](https://github.com/ssvb/tinymembench) - A simple memory benchmark tool, focusing on benchmarking memory bandwidth and latency with minimal dependencies, suitable for various platforms.
- [Geekbench](https://www.geekbench.com/) - Cross platform benchmarking tool
- [Empirical Roofline Tool (ERT)](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/software/ert/) - Create empirical roofline plots, alternative to intel vtune for any machine
- [Roofline Visualizer for ERT](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/software/roofline-visualizer/) - Visualizer for ERT
- [Caliper](https://github.com/LLNL/Caliper) - A Performance Analysis Toolbox in a Library
- [KDiskMark](https://github.com/JonMagon/KDiskMark) - Benchmarking Tool For SSD/HDD Drives
- [OpenBenchmarking](https://openbenchmarking.org/) - Open benchmarks on a variety of algorithms and hardware
- [Phoronix Test Suite](https://github.com/phoronix-test-suite/phoronix-test-suite) - Benchmarking suite for Linux

#### IO/Visualization Tools for HPC
- [ADIOS2](https://github.com/ornladios/ADIOS2) - The Adaptable IO System version 2, designed for flexible and efficient I/O for scientific data, supporting a wide range of HPC simulations.
- [Amira](https://www.thermofisher.com/ca/en/home/electron-microscopy/products/software-em-3d-vis/amira-software.html) - A powerful, multifaceted 3D software platform for visualizing, manipulating, and understanding Life Science and bio-medical data coming from all types of sources.
- [hdf5](https://www.hdfgroup.org/solutions/hdf5/) - The Hierarchical Data Format version 5 (HDF5), is an open source file format that supports large, complex, heterogeneous data.
- [paraview](https://www.paraview.org/) - An open-source, multi-platform data analysis and visualization application.
- [Scientific Visualization Wiki](https://en.wikipedia.org/wiki/Scientific_visualization) - A comprehensive guide to the field of scientific visualization, detailing techniques, tools, and applications.
- [the yt project](https://yt-project.org/) - An open-source, Python-based package for analyzing and visualizing volumetric data.
- [vedo](https://vedo.embl.es/) - A lightweight and powerful python module for scientific analysis and visualization of 3D objects and point clouds based on VTK.
- [visit](https://wci.llnl.gov/simulation/computer-codes/visit) - An Open Source, interactive, scalable, visualization, animation and analysis tool.

#### General Purpose Scientific Computing Libraries for HPC
 - [petsc](https://petsc.org/release/)
 - [ginkgo](https://ginkgo-project.github.io/)
 - [GSL](https://www.gnu.org/software/gsl/)
 - [Scalapack](https://netlib.org/scalapack/)
 - [rapids.ai - collection of libraries for executing end-to-end data science pipelines completely in the GPU](rapids.ai)
 - [trilinos](https://trilinos.github.io/)
 - [tnl project](https://tnl-project.org/)
 
#### Misc.
 - [mimalloc memory allocator](https://github.com/microsoft/mimalloc)
 - [jemalloc memory allocator](https://github.com/jemalloc/jemalloc)
 - [tcmalloc memory allocator](https://github.com/google/tcmalloc)
 - [Horde memory allocator](https://github.com/emeryberger/Hoard)
 - [Software utilization at UK National Supercomputing Service, ARCHER2](https://www.archer2.ac.uk/support-access/status.html#software-usage-data)
  
#### Wikis
- [Comparison of cluster software](https://en.wikipedia.org/wiki/Comparison_of_cluster_software)
- [List of cluster management software](https://en.wikipedia.org/wiki/List_of_cluster_management_software)

## Hardware

### Interconnects/Topology

- [Ethernet](https://en.wikipedia.org/wiki/Ethernet)
- [Infiniband](https://en.wikipedia.org/wiki/InfiniBand)
- [Network topologies](https://www.hpcwire.com/2019/07/15/super-connecting-the-supercomputers-innovations-through-network-topologies/)
- [Battle of the infinibands - Omnipath vs Infiniband](https://www.nextplatform.com/2017/11/29/the-battle-of-the-infinibands/)
- [Mellanox infiniband cluster config](https://www.mellanox.com/clusterconfig/)
- [RoCE - RDMA Over Converged Ethernet](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet)
- [Slingshot interconnect](https://www.hpe.com/ca/en/compute/hpc/slingshot-interconnect.html)
- [CXL - Compute Express Link](https://www.computeexpresslink.org/)
- [Infiniband Essentials](https://academy.nvidia.com/en/course/infiniband-essentials/?cm=244)
  
### CPU
- [Wikichip](https://en.wikichip.org/wiki/WikiChip)
- [Microarchitecture of Intel/AMD CPUs](https://www.agner.org/optimize/microarchitecture.pdf)
- [Apple M1](https://en.wikipedia.org/wiki/Apple_M1)
- [Apple M2](https://en.wikipedia.org/wiki/Apple_M2)
- [Apple M2 Teardown](https://www.ifixit.com/News/62674/m2-macbook-air-teardown-apple-forgot-the-heatsink)
- [Apply M1/M2 AMX](https://github.com/corsix/amx)
- [Apple M3](https://en.wikipedia.org/wiki/Apple_M3)
- [List of Intel processors](https://en.wikipedia.org/wiki/List_of_Intel_processors)
- [List of Intel micro architectures](https://en.wikipedia.org/wiki/List_of_Intel_CPU_microarchitectures)
- [Comparison of Intel processors](https://en.wikipedia.org/wiki/Comparison_of_Intel_processors)
- [Comparison of Apple processors](https://en.wikipedia.org/wiki/Apple-designed_processors)
- [List of AMD processors](https://en.wikipedia.org/wiki/List_of_AMD_processors)
- [List of AMD CPU micro architectures](https://en.wikipedia.org/wiki/List_of_AMD_CPU_microarchitectures)
- [Comparison of AMD architectures](https://en.wikipedia.org/wiki/Table_of_AMD_processors)

### GPU

- [Gpu Architecture Analysis](https://graphicscodex.courses.nvidia.com/app.html?page=_rn_parallel)
- [A trip through the Graphics Pipeline](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)
- [A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/)
- [Gentle Intro to GPU Inner Workings](https://vksegfault.github.io/posts/gentle-intro-gpu-inner-workings/)
- [AMD Instinct GPUs](https://en.wikipedia.org/wiki/AMD_Instinct_accelerators)
- [AMD GPU ROCm Support and OS Compatibility](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
- [List of AMD GPUs](https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units)
- [Comparison of CUDA architectures](https://en.wikipedia.org/wiki/CUDA)
- [Tales of the M1 GPU](https://asahilinux.org/2022/11/tales-of-the-m1-gpu/)
- [List of Intel GPUs](https://en.wikipedia.org/wiki/List_of_Intel_graphics_processing_units)
- [Performance of DGX Cluster](https://www.computer.org/csdl/proceedings-article/cloudcom/2022/636700a170/1JNqFu7QdTG)

### TPU/Tensor Cores

- [Google TPU](https://thechipletter.substack.com/p/googles-first-tpu-architecture)
- [TPU Wiki](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)

### Many integrated core processor (MIC)

- [Xeon Phi](https://en.wikipedia.org/wiki/Xeon_Phi)

### Cloud

- [Awesome Cloud HPC](https://github.com/kjrstory/awesome-cloud-hpc)

#### Vendors

- [AWS HPC](https://aws.amazon.com/hpc/)
- [Azure HPC](https://azure.microsoft.com/en-us/solutions/high-performance-computing/#intro)
- [rescale](https://rescale.com/)
- [vast.ai](https://vast.ai/)
- [vultr - cheap bare metal CPU, GPU, DGX servers](vultr.com)
- [hetzner - cheap servers incl. 80-core ARM](https://www.hetzner.com/)
- [Ampere ARM cloud-native processors](https://amperecomputing.com/)
- [Scaleway](https://www.scaleway.com/en/)
- [Chameleon Cloud](https://www.chameleoncloud.org/)
- [Lambda Labs](https://lambdalabs.com/)
- [Runpod](https://www.runpod.io/)
  
#### Articles/Papers
- [The use of Microsoft Azure for high performance cloud computing – A case study](https://www.diva-portal.org/smash/get/diva2:1704798/FULLTEXT01.pdf)
- [AWS Cluster in the cloud](https://cluster-in-the-cloud.readthedocs.io/en/latest/aws-infrastructure.html)
- [AWS Parallel Cluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorials-running-your-first-job-on-version-3.html)
- [AWS HPC Workshop](https://www.hpcworkshops.com/)
- [An Empirical Study of Containerized MPI and GUI Application on HPC in the Cloud](https://ieeexplore.ieee.org/abstract/document/10046607)

### Custom/FPGA/ASIC/APU

- [OpenPiton](http://parallel.princeton.edu/openpiton/)
- [Parallela](https://www.parallella.org/)
- [AMD APU](https://en.wikipedia.org/wiki/AMD_Accelerated_Processing_Unit)

### Certification

- [Intel Cluster Ready](https://en.wikipedia.org/wiki/Intel_Cluster_Ready)

### Student Opportunities / Workshops

- [Supercomputing Conference Student Opportunities](https://sc21.supercomputing.org/program/studentssc/)
- [SCC Student cluster competition](https://www.studentclustercompetition.us/)
- [Winter Classic Invitational](https://www.winterclassicinvitational.com/)
- [Linux Cluster Institute](https://linuxclustersinstitute.org/)

### Other/Wikis

- [Supercomputer](https://en.wikipedia.org/wiki/Supercomputer)
- [Supercomputer architecture](https://en.wikipedia.org/wiki/Supercomputer_architecture)
- [Beowulf cluster](https://en.wikipedia.org/wiki/Beowulf_cluster)
- [Computer cluster](https://en.wikipedia.org/wiki/Computer_cluster)
- [Comparison of Intel processors](https://en.wikipedia.org/wiki/Comparison_of_Intel_processors)
- [Comparison of Apple processors](https://en.wikipedia.org/wiki/Apple-designed_processors)
- [Comparison of AMD architectures](https://en.wikipedia.org/wiki/Table_of_AMD_processors)
- [Comparison of CUDA architectures](https://en.wikipedia.org/wiki/CUDA)
- [Cache](https://en.wikipedia.org/wiki/Cache_(computing))
- [Google TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)
- [IPMI](https://en.wikipedia.org/wiki/Intelligent_Platform_Management_Interface)
- [FRU](https://en.wikipedia.org/wiki/Field-replaceable_unit)
- [Disk Arrays](https://en.wikipedia.org/wiki/Disk_array)
- [RAID](https://en.wikipedia.org/wiki/RAID)
- [Cray](https://en.wikipedia.org/wiki/Cray)
- [Digital Signal Processors](https://en.wikipedia.org/wiki/Digital_signal_processor)
- [Vector Processor](https://en.wikipedia.org/wiki/Vector_processor)
  
## People

 - [Jack Dongarra - 2021 Turing Award - LINPACK, BLAS, LAPACK, MPI](https://www.nature.com/articles/s43588-022-00245-w)
 - [Bill Gropp - 2010 IEEE TCSC Medal for Excellence in Scalable Computing](https://en.wikipedia.org/wiki/Bill_Gropp)
 - [David Bader - built the first Linux supercomputer](https://en.wikipedia.org/wiki/David_Bader_(computer_scientist))
 - [Thomas Sterling - "Father of Beowulf clusters", ParalleX/HPX](https://en.wikipedia.org/wiki/Thomas_Sterling_(computing))
 - [Seymour Cray - Inventor of the Cray Supercomputer](https://en.wikipedia.org/wiki/Seymour_Cray)
 - [Larry Smarr - HPC Application Pioneer](https://en.wikipedia.org/wiki/Larry_Smarr)
 - [Donald Becker - Beowulf cluster software, Gordon Bell Prize Winner](https://en.wikipedia.org/wiki/Donald_Becker)
  
## Resources

#### Books/Manuals
- [Free Modern HPC Books by Victor Eijkhout](https://theartofhpc.com/)
- [High Performance Parallel Runtimes](https://www.amazon.com/High-Performance-Parallel-Runtimes-Implementation-ebook/dp/B08WH82KF9/ref=sr_1_1?keywords=High+Performance+Parallel+Runtimes&qid=1689287759&sr=8-1)
- [The OpenMP Common Core: Making OpenMP Simple Again](https://www.amazon.com/OpenMP-Common-Core-Engineering-Computation/dp/0262538865/ref=d_pd_sbs_sccl_2_1/130-5660046-7109016?pd_rd_w=Cqnxw&content-id=amzn1.sym.3676f086-9496-4fd7-8490-77cf7f43f846&pf_rd_p=3676f086-9496-4fd7-8490-77cf7f43f846&pf_rd_r=HG04QQS87WDHAGV578EE&pd_rd_wg=u0csS&pd_rd_r=8a6a0024-5dec-4934-8fa5-99e24d9fc4bd&pd_rd_i=0262538865&psc=1)
- [Parallel and High Performance Computing](https://www.manning.com/books/parallel-and-high-performance-computing)
- [Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/)
- [High Performance Computing: Modern Systems and Practices](https://www.amazon.ca/High-Performance-Computing-Systems-Practices/dp/012420158X) - Thomas Sterling, Maciej Brodowicz, Matthew Anderson 2017
- [Introduction to High Performance Computing for Scientists and Engineers](https://www.amazon.ca/Introduction-Performance-Computing-Scientists-Engineers/dp/143981192X/ref=sr_1_1?crid=1L276HPEB8K7I&keywords=Introduction+to+High+Performance+Computing+for+Scientists+and+Engineers&qid=1645137608&s=books&sprefix=introduction+to+high+performance+computing+for+scientists+and+engineers%2Cstripbooks%2C46&sr=1-1) - Hager 2010
- [Computer Organization and Design](https://www.amazon.ca/Computer-Organization-Design-RISC-V-Interface/dp/0128203315/ref=sr_1_1?crid=1XLX1HWLGRVO6&keywords=Computer+Organization+and+Design&qid=1645137443&s=books&sprefix=computer+organization+and+design%2Cstripbooks%2C48&sr=1-1)
- [Optimizing HPC Applications with Intel Cluster Tools: Hunting Petaflops](C+Applications+with+Intel+Cluster+Tools&qid=1645137507&s=books&sprefix=optimizing+hpc+applications+with+intel+cluster+tools%2Cstripbooks%2C80&sr=1-1)
- [Introduction to High Performance Scientific Computing](https://web.corral.tacc.utexas.edu/CompEdu/pdf/stc/EijkhoutIntroToHPC.pdf) - Victor Eijkhout 2021
- [Parallel Programming for Science and Engineering](https://web.corral.tacc.utexas.edu/CompEdu/pdf/pcse/EijkhoutParallelProgramming.pdf) - Victor EIjkhout 2021
- [Parallel Programming for Science and Engineering - HTML Version](https://pages.tacc.utexas.edu/~eijkhout/pcse/html/)
- [C++ High Performance](https://www.amazon.ca/High-Performance-Master-optimizing-functioning/dp/1839216549/ref=sr_1_1?crid=31OVX4VQ6Z84X&keywords=C%2B%2B+high+performance&qid=1640671313&sprefix=c%2B%2B+high+performance%2Caps%2C99&sr=8-1)
- [Data Parallel C++ Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL](https://www.apress.com/gp/book/9781484255735)
- [High Performance Python](https://www.amazon.ca/High-Performance-Python-Performant-Programming/dp/1449361595)
- [C++ Concurrency in Action: Practical Multithreading](https://www.manning.com/books/c-plus-plus-concurrency-in-action) - Anthony Williams 2012
- [The Art of Multiprocessor Programming](https://www.amazon.com/Art-Multiprocessor-Programming-Revised-Reprint/dp/0123973376/ref=sr_1_1?ie=UTF8&qid=1438003865&sr=8-1&keywords=maurice+herlihy) - Maurice Herlihy 2012
- [Parallel Computing: Theory and Practice](http://www.cs.cmu.edu/afs/cs/academic/class/15210-f15/www/tapp.html#ch:work-stealing) - Umut A. Acar 2016
- [Introduction to Parallel Computing](https://www.amazon.ca/Introduction-Parallel-Computing-Zbigniew-Czech/dp/1107174392/ref=sr_1_7?dchild=1&keywords=parallel+computing&qid=1625711415&sr=8-7) - Zbigniew J. Czech
- [Practical guide to bare metal C++](https://arobenko.github.io/bare_metal_cpp/)
- [Optimizing software in C++](https://www.agner.org/optimize/optimizing_cpp.pdf)
- [Optimizing subroutines in assembly code](https://www.agner.org/optimize/optimizing_assembly.pdf)
- [Microarchitecture of Intel/AMD CPUs](https://www.agner.org/optimize/microarchitecture.pdf)
- [Parallel Programming with MPI](https://www.cs.usfca.edu/~peter/ppmpi/)
- [HPC, Big Data, AI Convergence Towards Exascale: Challenge and Vision](https://www.taylorfrancis.com/books/edit/10.1201/9781003176664/hpc-big-data-ai-convergence-towards-exascale-olivier-terzo-jan-martinovi%C4%8D?refId=2cd8b0ad-d63d-42fa-9c3e-fe47fbbe0e29&context=ubx)
- [Introduction to parallel computing](https://www.amazon.com/Introduction-Parallel-Computing-Ananth-Grama/dp/0201648652/ref=sr_1_1?crid=LE1VD245VDX5&keywords=Ananth+Grama+-+Introduction+to+parallel+computing&qid=1644907263&sprefix=ananth+grama+-+introduction+to+parallel+computing%2Caps%2C43&sr=8-1) - Ananth Grama
- [The Student Supercomputer Challenge Guide](https://www.amazon.ca/Student-Supercomputer-Challenge-Guide-Supercomputing/dp/9811338310/ref=sr_1_1?crid=2J5374I76RP2Y&keywords=The+student+supercomputer+challenge&qid=1657060946&sprefix=the+student+supercomputer+challenge%2Caps%2C53&sr=8-1)
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/introduction.html)
- [E-Zines on Bash, Linux, Perf, etc - Julia Evans](https://wizardzines.com/)
- [The Art of Writing Efficient Programs: An Advanced Programmer's Guide to Efficient Hardware Utilization and Compiler Optimizations Using C++ Examples](https://www.amazon.ca/Art-Writing-Efficient-Programs-optimizations/dp/1800208111)
- [OpenMP Examples - openmp.org](https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf)
- [Latest books on OpemMP - openmp.org](https://www.openmp.org/resources/openmp-books/)
- [Programming Massively Parallel Processors 4th Edition 2023](https://www.amazon.ca/Programming-Massively-Parallel-Processors-Hands/dp/0128119861/ref=sr_1_1?crid=18EW0LVO2VFMC&keywords=Programming+Massively+Parallel+Processors+4th+Edition+2023&qid=1695110729&s=books&sprefix=programming+massively+parallel+processors+4th+edition+2023%2Cstripbooks%2C88&sr=1-1)
- [Software Optimization Cookbook](https://www.amazon.ca/Software-Optimization-Cookbook-Performance-Platforms/dp/0976483211)
- [Power and Performance_ Software Analysis and Optimization](https://www.amazon.ca/Power-Performance-Software-Analysis-Optimization-ebook/dp/B00WZ1AX6S/ref=sr_1_1?crid=22HMPRFCYAXC0&keywords=Power+and+Performance_+Software+Analysis+and+Optimization&qid=1695111518&s=books&sprefix=power+and+performance_+software+analysis+and+optimization%2Cstripbooks%2C85&sr=1-1)
- [Gropp books on MPI](https://wgropp.cs.illinois.edu/usingmpiweb/)
- [Performance Analysis and Tuning on Modern CPUs](https://book.easyperf.net/perf_book)
- [High Performance Computing in Biomimetics Modeling, Architecture and Applications](https://link.springer.com/book/10.1007/978-981-97-1017-1)
- [Systems Performance - Brendan Gregg](https://www.amazon.com/Systems-Performance-Brendan-Gregg/dp/0136820158)
- [Is Parallel Programming Hard, And, If So, What Can You Do About It? - Paul E. McKenney](https://cdn.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.html)
  
#### Courses
- [HPC Carpentry](https://www.hpc-carpentry.org/)
- [Berkeley: Applications of Parallel Computers](https://sites.google.com/lbl.gov/cs267-spr2019/) - Detailed course on HPC
- [CS6290 High-performance Computer Architecture](https://www.udacity.com/course/high-performance-computer-architecture--ud007) - Milos Prvulovic and Catherine Gamboa at George Tech
- [Udacity High Performance Computing](https://www.youtube.com/playlist?list=PLAwxTw4SYaPk8NaXIiFQXWK6VPnrtMRXC)
- [Parallel Numerical Algorithms](https://solomonik.cs.illinois.edu/teaching/cs554/index.html)
- [Vanderbilt - Intro to HPC](https://github.com/vanderbiltscl/SC3260_HPC)
- [Illinois - Intro to HPC](https://andreask.cs.illinois.edu/Teaching/HPCFall2012/) - Creator of PyCuda
- [Archer1 Courses](http://www.archer.ac.uk/training/past_courses.php)
- [TACC tutorials](https://portal.tacc.utexas.edu/tutorials)
- [Livermore training materials](https://hpc.llnl.gov/training/tutorials)
- [Xsede training materials](https://www.hpc-training.org/xsede/moodle/)
- [Parallel Computation Math](https://www.cct.lsu.edu/~pdiehl/teaching/2021/4997/)
- [Introduction to High-Performance and Parallel Computing - Coursera](https://www.coursera.org/learn/introduction-high-performance-computing)
- [Foundations of HPC 2020/2021](https://github.com/Foundations-of-HPC)
- [Principles of Distributed Computing](https://disco.ethz.ch/courses/podc_allstars/)
- [High Performance Visualization](https://www.uni-bremen.de/ag-high-performance-visualization)
- [Temple course on building/maintaining a cluster](https://www.hpc.temple.edu/mhpc/2021/hpc-technology/index.html)
- [Nvidia Deep Learning Course](https://www.nvidia.com/en-us/training/online/)
- [Coursera GPU Programming Specialization](https://www.coursera.org/specializations/gpu-programming)
- [Coursera Fundamentals of Parallelism on Intel Architecture](https://www.coursera.org/learn/parallelism-ia)
- [Coursera Introduction to High Performance Computing](https://www.coursera.org/learn/introduction-high-performance-computing)
- [Archer2 Shared Memory Programming with OpenMP](https://www.archer2.ac.uk/training/courses/210000-openmp-self-service/)
- [Archer2 Message-Passing Programming with MPI](https://www.archer2.ac.uk/training/courses/210000-mpi-self-service/)
- [HetSys 2022 Course](https://www.youtube.com/playlist?list=PL5Q2soXY2Zi9XrgXR38IM_FTjmY6h7Gzm)
- [Edukamu Introduction to Supercomputing](https://edukamu.fi/elements-of-supercomputing)
- [Heterogeneous Parallel Programming by S K](https://www.youtube.com/channel/UCbD5dhBi6DBSvCTgEDFz7uA/videos)
- [NCSA HPC Training Moodle](https://www.hpc-training.org/xsede/moodle/)
- [Supercomputing in plain english](http://www.oscer.ou.edu/education.php)
- [Cornell workshop](https://cvw.cac.cornell.edu/topics)
- [Carpentries Incubator HPC Intro](https://carpentries-incubator.github.io/hpc-intro/)
- [UL HPC School](https://ulhpc-tutorials.readthedocs.io/en/latest/hpc-school/)
- [Introduction to High-Performance Parallel Distributed Computing using Chapel, UPC++ and Coarray Fortran](https://bitbucket.org/berkeleylab/upcxx/wiki/events/CUF23)
- [Performance Engineering off Software Systems (MIT-OCW)](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/video_galleries/lecture-videos/)
- [Introduction to Parallel Computing (CMSC 498X/818X)](https://www.cs.umd.edu/class/fall2020/cmsc498x/lectures.shtml)
- [Infiniband Essentials](https://academy.nvidia.com/en/course/infiniband-essentials/?cm=244)
- [Performance Ninja Optimization Course](https://github.com/dendibakh/perf-ninja)
- [HPC Administration Virtual Residency 2024](https://www.youtube.com/@VirtualResidency2024/videos)
    
#### Tutorials/Guides/Articles
##### General
- [MpiTutorial](mpitutorial.com) - A fantastic mpi tutorial
- [Beginners Guide to HPC](http://www.shodor.org/petascale/materials/UPModules/beginnersGuideHPC/)
- [Rookie HPC Guide](https://rookiehpc.github.io/index.html)
- [RedHat High Performance Computing 101](https://www.redhat.com/en/blog/high-performance-computing-101)
- [Parallel Computing Training Tutorials](https://hpc.llnl.gov/training/tutorials) - Lawrence Livermore National Laboratory
- [Foundations of Multithreaded, Parallel, and Distributed Programming](https://www.amazon.com/Foundations-Multithreaded-Parallel-Distributed-Programming/dp/B00F4I7HM2/ref=sr_1_2?dchild=1&keywords=Gregory+R.+Andrews+Distributed+Programming&qid=1625766665&s=books&sr=1-2)
- [Building pipelines using slurm dependencies](https://hpc.nih.gov/docs/job_dependencies.html)
- [Writing slurm scripts in python,r and bash](https://vsoch.github.io/lessons/sherlock-jobs/)
- [Xsede new user tutorials](https://portal.xsede.org/online-training)
- [Supercomputing in plain english](http://www.oscer.ou.edu/education.php)
- [Improving Performance with SIMD intrinsics](https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/)
- [Want speed? Pass by value](https://web.archive.org/web/20140205194657/http://cpp-next.com/archive/2009/08/want-speed-pass-by-value/)
- [Introduction to low level bit hacks](https://catonmat.net/low-level-bit-hacks)
- [How to write fast numerical code: An Introduction](https://users.ece.cmu.edu/~franzf/papers/gttse07.pdf)
- [Lecture notes on Loop optimizations](https://www.cs.cmu.edu/~fp/courses/15411-f13/lectures/17-loopopt.pdf)
- [A practical approach to code optimization](https://www.einfochips.com/wp-content/uploads/resources/a-practical-approach-to-optimize-code-implementation.pdf)
- [Software optimization manuals](https://www.agner.org/optimize/)
- [Guide into OpenMP: Easy multithreading programming for C++](https://bisqwit.iki.fi/story/howto/openmp/)
- [An Introduction to the Partitioned Global Address Space (PGAS) Programming Model](https://cnx.org/contents/gtg1AzdI@7/An-Introduction-to-the-Partitioned-Global-Address-Space-PGAS-Programming-Model)
- [Jax in 2022](https://www.assemblyai.com/blog/why-you-should-or-shouldnt-be-using-jax-in-2022/)
- [C++ Benchmarking for beginners](https://unum.cloud/post/2022-03-04-gbench/)
- [Mapping MPI ranks to multiple cuda GPU](https://github.com/olcf-tutorials/local_mpi_to_gpu)
- [Oak Ridge National Lab Tutorials](https://github.com/olcf-tutorials)
- [How to perform large scale data processing in bioinformatics](https://medium.com/dnanexus/how-to-perform-large-scale-data-processing-in-bioinformatics-4006e8088af2)
- [Step by step SGEMM in OpenCL](https://cnugteren.github.io/tutorial/pages/page1.html)
- [Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- [Allocating large blocks of memory in bare-metal C programming](https://lemire.me/blog/2020/01/17/allocating-large-blocks-of-memory-bare-metal-c-speeds/)
- [Hashmap benchmarks 2022](https://martin.ankerl.com/2022/08/27/hashmap-bench-01/)
- [LLNL HPC Tutorials](https://hpc.llnl.gov/documentation/tutorials)
- [High Performance Computing: A Bird's Eye View](https://umashankar.blog/high-performance-computing-a-birds-eye-view/)
- [The dirty secret of high performance computing](https://www.techradar.com/news/the-dirty-secret-of-high-performance-computing)
- [Multiple GPUs with pytorch](https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained)
- [Brendan Gregg on Linux Performance](https://www.brendangregg.com/linuxperf.html)
- [Automatic Slurm build scripts](https://www.ni-sp.com/slurm-build-script-and-container-commercial-support/#h-automatic-slurm-build-script-for-rh-centos-7-8-and-9)
- [Fastest unordered_map implementation / benchmarks](https://martin.ankerl.com/2022/08/27/hashmap-bench-01/)
- [Memory bandwith NapkinMath](https://www.forrestthewoods.com/blog/memory-bandwidth-napkin-math/)
- [Avoiding Instruction Cache Misses](https://paweldziepak.dev/2019/06/21/avoiding-icache-misses/)
- [Multi-GPU Programming with Standard Parallel C++](https://developer.nvidia.com/blog/multi-gpu-programming-with-standard-parallel-c-part-1/)
- [EuroCC National Competence Center Sweden (ENCCS) HPC tutorials](https://enccs.se/lessons/)
- [LLNL hpc tutorials](https://hpc-tutorials.llnl.gov/)
- [python.org Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [HPC toolset tutorial (cluster management)](https://github.com/ubccr/hpc-toolset-tutorial)
- [OpenMP tutorials](https://www.openmp.org/resources/tutorials-articles/)
- [CUDA best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Understanding CPU Architecture And Performance Using LIKWID](https://pramodkumbhar.com/2020/03/architectural-optimisations-using-likwid-profiler/)
- [32 OpenMP Traps For C++ Developers](https://pvs-studio.com/en/blog/posts/cpp/a0054/#ID0EWEAC)
- [Best practices for running jobs on a HPC cluster](https://hpc.dccn.nl/docs/cluster_howto/best_practices.html)
- [Glossary of HPC related terms](https://www.gigabyte.com/Glossary?lan=en)
- [Setting the record straight: What is HPC?](https://www.gigabyte.com/Article/setting-the-record-straight-what-is-hpc-a-tech-guide-by-gigabyte?lan=en)
- [Atomic operations and contention](https://fgiesen.wordpress.com/2014/08/18/atomics-and-contention/)
- [A concurrency cost hiearchy](https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html)
  
##### Machine Learning Related
- [Best practices for machine learning with HPC](https://info.gwdg.de/news/en/best-practices-for-machine-learning-with-hpc/)
- [How to pick the right hardware for AI - Gigabyte - Part 1](https://www.gigabyte.com/Article/how-to-pick-the-right-server-for-ai-part-one-cpu-gpu)
- [A practitioner's guide to testing and running large GPU clusters for training generative AI models](https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters-for-training-generative-ai-models)
- [AWS HPC Workshop](https://www.hpcworkshops.com/)
- [Hardware Acceleration of LLMs: A comprehensive survey and comparison](https://news.ycombinator.com/item?id=41470074)
  
#### Review Papers/Articles
- [Interactive and Urgent HPC Challenges (2024)](https://arxiv.org/pdf/2401.14550.pdf)
- [The Landscape of Exascale Research: A Data-Driven Literature Analysis (2020)](https://dl.acm.org/doi/pdf/10.1145/3372390)
- [The Landscape of Parallel Computing Research: A View from Berkeley](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf)
- [Extreme Heterogeneity 2018: Productive Computational Science in the Era of Extreme Heterogeneity](references/2018-Extreme-Heterogeneity-DoE.pdf)
- [Programming for Exascale Computers - Will Gropp, Marc Snir](https://snir.cs.illinois.edu/listed/J55.pdf)
- [On the Memory Underutilization: Exploring Disaggregated Memory on HPC Systems (2020)](https://www.mcs.anl.gov/research/projects/argo/publications/2020-sbacpad-peng.pdf)
- [Advances in Parallel & Distributed Processing, and Applications (conference proceedings)](https://link.springer.com/book/10.1007/978-3-030-69984-0)
- [Designing Heterogeneous Systems: Large Scale Architectural Exploration Via Simulation](https://ieeexplore.ieee.org/abstract/document/9651152)
- [Reinventing High Performance Computing: Challenges and Opportunities (2022)](https://arxiv.org/pdf/2203.02544.pdf)
- [Challenges in Heterogeneous HPC White Paper (2022)](https://www.etp4hpc.eu/pujades/files/ETP4HPC_WP_Heterogeneous-HPC_20220216.pdf)
- [An Evolutionary Technical & Conceptual Review on High Performance Computing Systems (Dec 2021)](https://kalaharijournals.com/resources/DEC_597.pdf)
- [New Horizons for High-Performance Computing (2022)](https://csdl-downloads.ieeecomputer.org/mags/co/2022/12/09963771.pdf?Expires=1669702667&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jc2RsLWRvd25sb2Fkcy5pZWVlY29tcHV0ZXIub3JnL21hZ3MvY28vMjAyMi8xMi8wOTk2Mzc3MS5wZGYiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2Njk3MDI2Njd9fX1dfQ__&Signature=s3K~-JXyED6vMVT9IKGj7LOhR75CrkQXiqAEsAEQt4zRqTbUFywmSoT10th1CdAaZcfZFuMsg23o2e719FRkCD6flVNB55d5tKyMUp7jUbkUtxnatOWLAKXfE4yQ-zrYQQEWBhtpSLKrTAS1oVmJ00YwkWqLYqCjhFIjW9La5od2SGQZEFZ136bbaGzxLZlED3JlMCMLB54YXKr-Ng1rngV4I9Wi-wSTFyLiA92~fUlk1KPQKU0XjtsMyYMYlt06Ze5H6jcQw4ytJ6c7r7qNJ43ifnsZepWmBywA8lVy2g3joOvZJtVjl~S91R8EZbiyWlYdWBGrO7pPdO6hH48~NQ__&Key-Pair-Id=K12PMWTCQBDMDT)
- [CConfidential High-Performance Computing in the Public Cloud](https://arxiv.org/pdf/2212.02378.pdf)
- [Containerisation for High Performance Computing Systems: Survey and Prospects](https://ieeexplore.ieee.org/abstract/document/9985426)
- [Heterogeneous Computing Systems (2023)](https://arxiv.org/pdf/2212.14418.pdf)
- [Myths and Legends in High-Performance Computing](https://arxiv.org/pdf/2301.02432.pdf)
- [Energy-Aware Scheduling for High-Performance Computing Systems: A Survey](https://www.mdpi.com/1996-1073/16/2/890)
- [Ultimate Physical limits to computation - Seth Lloyd](https://arxiv.org/abs/quant-ph/9908043)
- [Myths and Legends in High-Performance Computing](https://arxiv.org/abs/2301.02432)
- [Abstract Machine Models and Proxy Architectures for Exascale Computing, 2014, Sandia National Laboratories and Lawrence Berkeley National Laboratory](https://www.osti.gov/servlets/purl/1561498)
- [Some thoughts on the environmental impact of High Performance Computing](https://sifflez.org/publications/environment-hpc/)
- [A Research Retrospective on AMD's Exascale Computing Journey](https://dl.acm.org/doi/abs/10.1145/3579371.3589349)
  
#### News
- [InsideHPC](https://insidehpc.com/)
- [HPCWire](https://www.hpcwire.com/)
- [NextPlatform](https://www.nextplatform.com)
- [Datacenter Dynamics](https://www.datacenterdynamics.com/en/)
- [Admin Magazine HPC](https://www.admin-magazine.com/HPC/News)
- [Toms hardware](https://www.tomshardware.com/)
- [Tech Radar](https://www.techradar.com/)
- [Phoronix](https://www.phoronix.com/)
- [The Register](https://www.theregister.com/on_prem/hpc/)

#### Podcasts
- [This week in HPC](https://soundcloud.com/this-week-in-hpc)
- [Preparing Applications for Aurora in the Exascale Era](https://connectedsocialmedia.com/20114/preparing-applications-for-aurora-in-the-exascale-era/)
- [Slurm podcast](https://www.rce-cast.com/index.php/Podcast/rce-10-slurm.html)
- [HPCPodcast](https://insidehpc.com/category/resources/hpc-podcast/)
- [Developer Stories - The path to a career in high performance computing is not always equitable or clear.](https://rseng.github.io/devstories/2024/jay-lofstead/)
- [Developer Stories - HPCToolkit](https://rseng.github.io/devstories/2024/wileam-phan/)
  
#### Video Presentations/Courses/Channels
- [Argonne lectures on Extreme Scale Computing 2022](https://www.youtube.com/playlist?list=PLcbxjEfgjpO9OeDu--H9_XqyxPj3MkjdN)
- [Argonne supercomputer tour](https://www.youtube.com/watch?v=UT9HCgp2X3A)
- [Containers in HPC - what they fix and what they break ](https://youtube.com/watch?v=WQTrA4-9ZXk&feature=share) 
- [HPC Tech Shorts](https://www.youtube.com/channel/UChSIn5kcWQvJxW17KIjdLVw)
- [CppCon](https://www.youtube.com/user/CppCon/videos)
- [Create a clustering server](https://www.youtube.com/watch?v=4LyL4sNZ1u4)
- [Argonne national lab](https://www.youtube.com/channel/UCfwgjtIQB3puojz_N9ly_Ag)
- [Oak Ridge National Lab](https://www.youtube.com/user/OakRidgeNationalLab)
- [Concurrency in C++20 and Beyond](https://www.youtube.com/watch?v=jozHW_B3D4U) - A. Williams
- [Is Parallel Programming still Hard?](https://www.youtube.com/watch?v=YM8Xy6oKVQg) - P. McKenney, M. Michael, and M. Wong at CppCon 2017
- [The Speed of Concurrency: Is Lock-free Faster?](https://www.youtube.com/watch?v=9hJkWwHDDxs) - Fedor G Pikus in CppCon 2016
- [Expressing Parallelism in C++ with Threading Building Blocks](https://www.youtube.com/watch?v=9Otq_fcUnPE) - Mike Voss at Intel Webinar 2018
- [A Work-stealing Runtime for Rust](https://www.youtube.com/watch?v=4DQakkJ8XLI) - Aaron Todd in Air Mozilla 2017
- [C++11/14/17 atomics and memory model: Before the story consumes you](https://www.youtube.com/watch?v=DS2m7T6NKZQ) - Michael Wong in CppCon 2015
- [The C++ Memory Model](https://www.youtube.com/watch?v=gpsz8sc6mNU) - Valentin Ziegler at C++ Meeting 2014
- [Sharcnet HPC](https://www.youtube.com/channel/UCCRmb5_GMWT2hSlALHlwIMg)
- [Low Latency C++ for fun and profit](https://www.youtube.com/watch?v=BxfT9fiUsZ4)
- [scalane python profiler](https://youtu.be/5iEf-_7mM1k)
- [Kokkos lectures](https://www.youtube.com/watch?v=rUIcWtFU5qM&t=698s)
- [EasyBuild Tech Talk I - The ABCs of Open MPI, part 1 (by Jeff Squyres & Ralph Castain)](https://www.youtube.com/watch?v=WpVbcYnFJmQ)
- [The Spack 2022 Roadmap](https://www.youtube.com/watch?v=HyA7RpjoY1k)
- [A Not So Simple Matter of Software | Talk by Turing Award Winner Prof. Jack Dongarra](https://youtu.be/QBCX3Oxp3vw)
- [Vectorization/SIMD intrinsics](https://www.youtube.com/watch?v=x9Scb5Mku1g)
- [New Silicon for Supercomputers: A Guide for Software Engineers](https://www.youtube.com/watch?v=w3xNLj6nRgs&t=197s)
- [TechTechPotato Channel](TechTechPotato)
- [How to write the perfect hash table ](https://www.youtube.com/watch?v=DMQ_HcNSOAI)
- [FosDem 2024 HPC Big Data Conference videos](https://fosdem.org/2024/schedule/track/hpc-big-data-data-science/)
- [Bright Computing Cluster Management Technical Overview](https://www.youtube.com/watch?v=0AxzcZuviW0)
- [What is HPC? An introduction by Canonical](https://www.youtube.com/watch?v=tGIobcyKViI)
- [Slurm job schedular basics](https://www.youtube.com/watch?v=Juo_mb3otJ0)
- [EasyBuild Tech Talk I - The ABCs of Open MPI, part 1 (by Jeff Squyres & Ralph Castain)](https://youtu.be/WpVbcYnFJmQ?feature=shared)
  
#### Presentation Slides
- [Task based Parallelism and why it's awesome](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Workshops/Conferences/CSAM-2015/Programme/lecture7a_gonnet-pdf.pdf?__blob=publicationFile) - Pedro Gonnet
- [Tuning Slurm Scheduling for Optimal Responsiveness and Utilization](https://slurm.schedmd.com/SUG14/sched_tutorial.pdf)
- [Parallel Programming Models Overview (2020)](https://www.researchgate.net/publication/348187154_Parallel_programming_models_overview_2020)
- [Comparative Analysis of Kokkos and Sycl (Jeff Hammond)](https://www.iwocl.org/wp-content/uploads/iwocl-2019-dhpcc-jeff-hammond-a-comparitive-analysis-of-kokkos-and-sycl.pdf) 
- [Hybrid OpenMP/MPI Programming](https://www.nersc.gov/assets/Uploads/NUG2013hybridMPIOpenMP2.pdf)
- [Designs, Lessons and Advice from Building Large Distributed Systems - Jeff Dean (Google)](http://www.cs.cornell.edu/projects/ladis2009/talks/dean-keynote-ladis2009.pdf)
- [Practical Debugging and Performance Engineering](https://orbilu.uni.lu/bitstream/10993/55305/1/Practical_Debugging_and_Performance_Engineering_for_HPC.pdf)

  
#### Building Clusters/Virtual Clusters
- [Resources for learning about HPC networks and storage r/HPC](https://www.reddit.com/r/HPC/comments/17o0q5d/resources_for_learning_about_hpc_networks_and/)
- [Slurm for dummies guide](https://github.com/SergioMEV/slurm-for-dummies)
- [Build a cluster under 50k](https://www.reddit.com/r/HPC/comments/srssrt/build_a_minicluster_under_50000/)
- [Build a Beowulf cluster](https://github.com/darshanmandge/Cluster) 
- [Build a Raspberry Pi Cluster](https://www.raspberrypi.com/tutorials/cluster-raspberry-pi-tutorial/)
- [Puget Systems](https://www.pugetsystems.com/)
- [Lambda Systems](https://lambdalabs.com/)
- [Titan computers](https://www.titancomputers.com)
- [Temple course on building/maintaining a cluster](https://www.hpc.temple.edu/mhpc/2021/hpc-technology/index.html)
- [Detailed reddit discussion on setting up a small cluster](https://www.reddit.com/r/HPC/comments/xeipt7/setting_up_a_small_hpc_cluster/)
- [Tiny titan - build a really cool pi supercomputer](https://github.com/tinytitan)
- [Building an Intel HPC cluster with OpenHPC](https://cdrdv2-public.intel.com/671501/installguide-openhpc2-centos8-18jul21.pdf)
- [Reddit r/HPC post on building clusters](https://www.reddit.com/r/HPC/comments/11azmhy/wanting_to_setup_a_cluster/)
- [Build a virtual cluster with PelicanHPC](https://sourceforge.net/projects/pelicanhpc/)
- [Building a High-performance Computing Cluster Using FreeBSD](https://people.freebsd.org/~brooks/papers/bsdcon2003/fbsdcluster/)
- [Supermicro GPU racks](https://www.supermicro.com/en/products/gpu)
- [VirtualOrfeo - Virtual HPC Cluster](https://gitlab.com/area7/datacenter/codes/virtualorfeo)
- [Is there a reason to build a raspberry pi clluster](https://www.reddit.com/r/HPC/comments/1bfywk8/is_there_ever_a_reason_to_build_a_raspberry_pi/)
    
#### Forums
 - [r/hpc](https://www.reddit.com/r/HPC/)
 - [r/homelab](https://www.reddit.com/r/homelab/)
 - [r/slurm](https://www.reddit.com/r/SLURM/)

#### Careers
 - [HPC University Careers search](http://hpcuniversity.org/careers/)
 - [HPC wire career site](https://careers.hpcwire.com/)
 - [HPC certification](https://www.hpc-certification.org/)
 - [HPC SysAdmin Jobs (reddit)](https://www.reddit.com/r/HPC/comments/w5eu66/systems_administrator_systems_engineer_jobs/)
 - [The United States Research Software Engineer Association](https://us-rse.org/)
 - [NCSA Internship](https://wiki.ncsa.illinois.edu/display/NCSACIP/NCSA+Internship+Program+for+CI+Professionals+Home)
 - [AI and Future HPC Job Prospect](https://www.reddit.com/r/HPC/comments/12anrgq/hpc_future_career_prospects/)
 - [HPC sys admin career (reddit)](https://www.reddit.com/r/HPC/comments/16jkqlv/it_support_for_an_academic_hpc_cluster_as_a_career/)
   
#### Membership Clubs
 - [Association for Computing Machinery](acm.org)
 - [ETP4HPC](https://www.etp4hpc.eu/)
 - [The SIGHPC Systems Professionals](https://sighpc-syspros.org/)
   
#### Blogs
 - [1024 Cores](http://www.1024cores.net/) - Dmitry Vyukov 
 - [The Black Art of Concurrency](https://www.internalpointers.com/post-group/black-art-concurrency) - Internal Pointers
 - [Cluster Monkey](https://www.clustermonkey.net/)
 - [Johnathon Dursi](https://www.dursi.ca/)
 - [Arm Vendor HPC blog](https://community.arm.com/developer/tools-software/hpc/b/hpc-blog)
 - [HPC Notes](https://www.hpcnotes.com/)
 - [Brendan Gregg Performance Blog](https://www.brendangregg.com/blog/index.html)
 - [Performance engineering blog](https://pramodkumbhar.com)
 - [Concurrency Freaks](https://concurrencyfreaks.blogspot.com/)
 - [Servers@Home](https://servers.hydrology.cc/blog/)
 - [Dr.Bandwith Blog](https://sites.utexas.edu/jdm4372/2010/10/01/welcome-to-dr-bandwidths-blog/)
 - [Johnny's Software Lab](https://johnnysswlab.com/)
 - [Daniel Lemire Blog](https://lemire.me/blog/)
 - [Gigabyte HPC Blog](https://www.gigabyte.com/)
   
#### Journals
 - [IEEE Transactions on Parallel and Distributed Systems (TPDS)](https://www.computer.org/csdl/journal/td) 
 - [Journal of Parallel and Distributed Computing](https://www.journals.elsevier.com/journal-of-parallel-and-distributed-computing)
  
#### Conferences

 - [ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP)](https://ppopp19.sigplan.org/home)
 - [ACM Symposium on Parallel Algorithms and Architectures (SPAA)](https://spaa.acm.org/)
 - [SC conference (SC)](https://supercomputing.org/)
 - [IEEE International Parallel and Distributed Processing Symposium (IPDPS)](http://www.ipdps.org/)
 - [International Conference on Parallel Processing (ICPP)](https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/)
 - [IEEE High Performance Extreme Computing Conference (HPEC)](https://ieee-hpec.org/cfp.htm)
 - [FosDem](https://fosdem.org/)

#### Communities/Chat Groups
  - [HPC Social Discord server](https://hpc.social/projects/chat/)
  - [HPC Social slack group](https://hpcsocial.slack.com/)
  - [HPC Social](https://hpc.social/)
  - [Beowulf Mailing List](https://www.beowulf.org/)
  - [Society of Research Software Engineering](https://society-rse.org/get-involved/)
  - [Women In HPC](https://womeninhpc.org/)
  - [HPC Hallway](https://hpc-hallway.github.io/The-Hallway/)
  - [The High Performance Computing Special Interest Group](https://hpc-sig.org.uk/)
  - [SigHPC](https://www.sighpc.org/)
    
#### Twitters
 - [Top500](https://twitter.com/top500supercomp?s=20)
 - [HPE HPC](https://twitter.com/hpe_hpc)
 - [HPC Wire](https://twitter.com/HPCwire)
 - [Rookie HPC](https://twitter.com/RookieHPC?s=20)
 - [HPC_Guru](https://twitter.com/HPC_Guru?s=20&t=jHjVtUaZhz4s6Rq62IAmYg)
 - [Jeff Hammond](https://twitter.com/science_dot)
 
#### Consulting
- [Redline Performance](https://redlineperf.com/)
- [R systems](http://rsystemsinc.com/)
- [Advanced Clustering](https://www.advancedclustering.com/)

#### Interview Preparation
  - [Reddit Entry Level HPC interview help](https://www.reddit.com/r/HPC/comments/nhpdfb/entrylevel_hpc_job_interview/)

#### Organizations
  - [Prace](https://prace-ri.eu/)
  - [Xsede](https://www.xsede.org/)
  - [Compute Canada](https://www.computecanada.ca/)
  - [Riken CSS](https://www.riken.jp/en/research/labs/r-ccs/)
  - [Pawsey](https://pawsey.org.au/)
  - [International Data Corporation](https://www.idc.com/)
  - [List of Federally funded research and development centers](https://en.wikipedia.org/wiki/Federally_funded_research_and_development_centers)

#### Interesting r/HPC posts
  - [finding a supercomputer to use for research](https://www.reddit.com/r/HPC/comments/19e58z7/how_do_i_go_about_finding_a_supercomputer_to_use/)

#### Misc. Wikis
- [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law)
- [HPC Wiki](https://hpc-wiki.info/hpc/HPC_Wiki)
- [FLOPS](https://en.wikipedia.org/wiki/FLOPS)
- [Computational complexity of math operations](https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations)
- [Many Task Computing](https://en.wikipedia.org/wiki/Many-task_computing)
- [High Throughput Computing](https://en.wikipedia.org/wiki/High-throughput_computing)
- [Parallel Virtual Machine](https://en.wikipedia.org/wiki/Parallel_Virtual_Machine)
- [OSI Model](https://en.wikipedia.org/wiki/OSI_model)
- [Workflow management](https://en.wikipedia.org/wiki/Scientific_workflow_system)
- [Compute Canada Documentation](https://docs.computecanada.ca/wiki/Compute_Canada_Documentation)
- [Network Interface Controller (NIC)](https://en.wikipedia.org/wiki/Network_interface_controller)
- [Just in time compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation)
- [List of distributed computing projects](https://en.wikipedia.org/wiki/List_of_distributed_computing_projects)
- [Computer cluster](https://en.wikipedia.org/wiki/Computer_cluster)
- [Quasi-opportunistic supercomputing](https://en.wikipedia.org/wiki/Quasi-opportunistic_supercomputing)
- [Limits of Computation](https://en.wikipedia.org/wiki/Limits_of_computation)
- [Bremermann's Limit](https://en.wikipedia.org/wiki/Bremermann%27s_limit)
- [Concurrency patterns](https://en.wikipedia.org/wiki/Concurrency_pattern)
- [Parallel Computing](https://en.wikipedia.org/wiki/Parallel_computing)
- [Server Management](https://wiki.hydrology.cc/en/home)
  
#### Misc. Papers/Articles
- [Advanced Parallel Programming in C++](https://www.diehlpk.de/assets/modern_cpp.pdf)
- [Tools for scientific computing](https://arxiv.org/pdf/2108.13053.pdf)
- [Quantum Computing for High Performance Computing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9537178)
- [Benchmarking data science: Twelve ways to lie with statistics and performance on parallel computers.](http://ww.unixer.de/publications/img/hoefler-12-ways-data-science-preprint.pdf)
- [Establishing the IO500 Benchmark](https://www.vi4io.org/_media/io500/about/io500-establishing.pdf)
- [NVIDIA High Performance Computing articles](https://research.nvidia.com/research-area/high-performance-computing)
- [Let's write a superoptimizer](https://austinhenley.com/blog/superoptimizer.html)
- [Why I think C++ is still a desirable coding platform compared to Rust](https://lucisqr.substack.com/p/why-i-think-c-is-still-a-very-attractive)
- [The State of Fortran (arxiv paper 2022)](https://arxiv.org/abs/2203.15110)
- [50 years later, is two phase locking still the best](https://concurrencyfreaks.blogspot.com/2023/09/50-years-later-is-two-phase-locking.html)
- [Estimating your memory bandwith](https://lemire.me/blog/2024/01/13/estimating-your-memory-bandwidth/)
  
#### Misc. Repos
  - [Build a Beowulf cluster](https://github.com/darshanmandge/Cluster)
  - [libsc - Supercomputing library](https://github.com/cburstedde/libsc)
  - [xbyak jit assembler](https://github.com/herumi/xbyak)
  - [cpufetch - pretty cpu info fetcher](https://github.com/Dr-Noob/cpufetch)
  - [RRZE-HPC](https://github.com/RRZE-HPC)
  - [Argonne Github](https://github.com/Argonne-National-Laboratory)
  - [Argonne Leadership Computing Facility](https://github.com/argonne-lcf)
  - [Oak Ridge National Lab Github](https://github.com/ORNL)
  - [Compute Canada](https://github.com/ComputeCanada)
  - [HPCInfo by Jeff Hammond](https://github.com/jeffhammond/HPCInfo)
  - [Texas Advanced Computing Center (TACC) Github](https://github.com/TACC)
  - [LANL HPC Github](https://github.com/hpc)
  - [Rust in HPC](https://github.com/westernmagic/rust-in-hpc)
  - [University of Buffalo - Center for Computational Research](https://github.com/ubccr)
  - [Center for High Performance Computing - University of Utah](https://github.com/CHPC-UofU)
  
#### Misc. Theses
   - [Rust programming language in the high-performance computing environment](https://www.research-collection.ethz.ch/handle/20.500.11850/474922)

#### Misc.
  - [Exascale Project](https://www.exascaleproject.org/)
  - [Pocket HPC Survival Guide](https://tin6150.github.io/psg/lsf.html)
  - [HPC Summer school](https://www.ihpcss.org/)
  - [Overview of all linear algebra packages](http://www.netlib.org/utk/people/JackDongarra/la-sw.html)
  - [Latency numbers](http://norvig.com/21-days.html#answers)
  - [Nvidia HPC benchmarks](https://ngc.nvidia.com/catalog/containers/nvidia:hpc-benchmarks)
  - [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#)
  - [AWS Cloud calculator](https://calculator.aws/)
  - [Quickly benchmark C++ functions](https://quick-bench.com/)
  - [LLNL Software repository](https://software.llnl.gov/)
  - [Boinc - volunteer computing projects](https://boinc.berkeley.edu/projects.php)
  - [Prace Training Events](https://events.prace-ri.eu/category/2/)
  - [Nice discussion on FlameGraph profiling](https://stackoverflow.com/questions/27842281/unknown-events-in-nodejs-v8-flamegraph-using-perf-events/27867426#27867426)
  - [Nice discussion on parts of a supercomputer on reddit](https://www.reddit.com/r/HPC/comments/11elh93/job_node_socket_task_runner_device_thread_logical/)
  - [Technical Report on C++ performance](https://www.open-std.org/jtc1/sc22/wg21/docs/TR18015.pdf)
  - [BOINC Compute for science](https://boinc.berkeley.edu/)
  - [Count prime numbers using MPI](https://people.sc.fsu.edu/~jburkardt/c_src/prime_mpi/prime_mpi.html)
  - [How to build your LEGO Scafell Pike Supercomputer](https://www.youtube.com/watch?v=m499o5rLh38)

#### Games/Challenges
  - [Deadlock empire - practice concurrency](https://github.com/deadlockempire/deadlockempire.github.io)
  - [Sad Server - practice linux server management](https://sadservers.com/scenarios)
    
## Other Curated Lists 
   - [Awesome Cloud HPC](https://github.com/kjrstory/awesome-cloud-hpc)
   - [Parallel Computing Guide](https://github.com/mikeroyal/Parallel-Computing-Guide)
   - [Awesome Parallel Computing](https://github.com/taskflow/awesome-parallel-computing)
   - [Princeton resources on OpenMP](https://researchcomputing.princeton.edu/education/external-online-resources/openmp)
   - [Awesome HPC](https://github.com/dstdev/awesome-hpc/)
   - [Sig HPC Education](https://sighpceducation.acm.org/resources/hpcresources/)
   - [Fortran Codes On Github](https://github.com/Beliavsky/Fortran-code-on-GitHub)
   - [Fortran Tools](https://github.com/Beliavsky/Fortran-Tools)
     
## Acknowledgements

This repo started from the great curated list https://github.com/taskflow/awesome-parallel-computing


