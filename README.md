<p align="center">
<img src="https://www.montana.edu/uit/rci/assets/hpc.png" width="600">
</p>

<p align="center">
A curated list of awesome high performance computing resources. 
</p>

## Table of Contents

 - [General Info](#general-info)
 - [Software](#software)
 - [Hardware](#hardware) 
 - [People](#people)
 - [Resources](#resources)
 - [Acknowledgements](#acknowledgements )

## General Info

### A Few Upcoming Supercomputers 
 - [El Capitan](https://www.amd.com/en/products/exascale-era) - 2023, AMD-based, ~1.5 exaflops
 - [Aurora](https://en.wikipedia.org/wiki/Aurora_(supercomputer)) - 2022, Intel-based, ~1 exaflops
 - [Tianhe-3](https://www.nextplatform.com/2019/05/02/china-fleshes-out-exascale-design-for-tianhe-3/) - 2022, ~700 Petaflop (Linpack500)

### Most Recent List of the Top500 Supercomputers
 - [Top500 (November 2022)](https://www.top500.org/lists/top500/2022/11/)
 - [Green500 (November 2022)](https://www.top500.org/lists/green500/2022/11/)
 - [io500](https://io500.org/)
 
### History
 - [History of Supercomputing (Wikipedia)](https://en.wikipedia.org/wiki/History_of_supercomputing)
 - [History of the Top500 (Wikipedia)](https://en.wikipedia.org/wiki/TOP500)

## Software

#### Popular HPC Programming Libraries/APIs/Tools/Standards

- [CAF: An Open Source Implementation of the Actor Model in C++](https://github.com/actor-framework/actor-framework)
- [Chapel: A Programming Language for Productive Parallel Computing on Large-scale Systems](https://chapel-lang.org/)
- [Charm++: Parallel Programming with Migratable Objects](http://charm.cs.illinois.edu/research/charm) 
- [Cilk Plus: C/C++ Extension for Data and Task Parallelism](https://www.cilkplus.org/)
- [Taskflow: A Modern C++ Parallel Task Programming Library](https://github.com/taskflow/taskflow)
- [FastFlow: High-performance Parallel Patterns in C++](https://github.com/fastflow/fastflow)
- [Galois: A C++ Library to Ease Parallel Programming with Irregular Parallelism](https://github.com/IntelligentSoftwareSystems/Galois)
- [Heteroflow: Concurrent CPU-GPU Task Programming using Modern C++](https://github.com/Heteroflow/Heteroflow)
- [HPX: A C++ Standard Library for Concurrency and Parallelism](https://github.com/STEllAR-GROUP/hpx)
- [Intel TBB: Threading Building Blocks](https://www.threadingbuildingblocks.org/)
- [Kokkos: A C++ Programming Model for Writing Performance Portable Applications on HPC platforms](https://github.com/kokkos/kokkos)
- [OpenMP: Multi-platform Shared-memory Parallel Programming in C/C++ and Fortran](https://www.openmp.org/)
- [RaftLib: A C++ Library for Enabling Stream and Dataflow Parallel Computation](https://github.com/RaftLib/RaftLib) 
- [STAPL: Standard Template Adaptive Parallel Programming Library in C++](https://parasol.tamu.edu/stapl/)
- [STLab: High-level Constructs for Implementing Multicore Algorithms with Minimized Contention](http://stlab.cc/libraries/concurrency/)
- [Transwarp: A Header-only C++ Library for Task Concurrency](https://github.com/bloomen/transwarp)
- [MPI: Message passing interface; OpenMPI implementation](https://www.open-mpi.org/)
- [MPI: Message passing interface: MPICH implementation](https://www.mpich.org/)
- [PVM: Parallel Virtual Maschine: A predecessor to MPI for distributed computing](https://www.csm.ornl.gov/pvm/)
- [CUDA: High performance NVIDIA GPU acceleration](https://developer.nvidia.com/cuda-toolkit)
- [OpenACC: "OpenMP for GPUs"](https://www.openacc.org/)
- [numba: Numba is an open source JIT compiler that translates a subset of Python into fast machine code.](https://numba.pydata.org/)
- [dask: Dask provides advanced parallelism for analytics, enabling performance at scale for the tools you love](https://dask.org)
- [ray: scale AI and Python workloads — from reinforcement learning to deep learning](https://www.ray.io/)
- [RAJA: architecture and programming model portability for HPC applications](https://github.com/LLNL/RAJA)
- [ROCM: first open-source software development platform for HPC/Hyperscale-class GPU computing](https://rocmdocs.amd.com/en/latest/)
- [HIP: HIP is a C++ Runtime API and Kernel Language for AMD/Nvidia GPU](https://github.com/ROCm-Developer-Tools/HIP)
- [MOGSLib - User defined schedulers](https://github.com/ECLScheduling/MOGSLib)
- [SYCL - C++ Abstraction layer for heterogeneous devices](https://www.khronos.org/sycl/)
- [Legion - Distributed heterogenous programming librrary](https://github.com/StanfordLegion/legion)
- [SkelCL – A Skeleton Library for Heterogeneous Systems](https://skelcl.github.io/)
- [Legate - Nvidia replacement for numpy based on Legion](https://github.com/nv-legate/legate.numpy)
- [The Open Community Runtime - Specification for Asynchronous Many Task systems](https://wiki.modelado.org/Open_Community_Runtime)
- [Pyfi - distributed flow and computation system](https://github.com/radiantone/pyfi)
- [HPC-X - Nvidia implementation of MPI](https://developer.nvidia.com/networking/hpc-x)
- [MPAVICH - Implementation of MPI](https://mvapich.cse.ohio-state.edu/)
- [mpi4py - python bindings for MPI](https://mpi4py.readthedocs.io/en/stable/)
- [UCX - optimized production proven-communication framework](https://github.com/openucx/ucx#using-ucx)
- [Horovod - distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet](https://github.com/horovod/horovod)
- [Taichi - parallel programming language for high-performance numerical computations (embedded in Python with JIT support)](https://github.com/taichi-dev/taichi)
- [MAGMA - next generation linear algebra (LA) GPU accelerated libraries](https://developer.nvidia.com/magma)
- [NVIDIA cuNumeric - GPU drop-in for numpy](https://developer.nvidia.com/cunumeric)
- [Halide - a language for fast, portable computation on images and tensors](https://halide-lang.org/index.html#gettingstarted)
- [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- [PMIX](https://pmix.github.io/standard)
- [NCCL - The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking](https://developer.nvidia.com/nccl)
- [Kompute - The general purpose GPU compute framework for cross vendor graphics cards (AMD, Qualcomm, NVIDIA & friends)](https://github.com/KomputeProject/kompute)
- [alpaka - The alpaka library is a header-only C++17 abstraction library for accelerator development](https://github.com/alpaka-group/alpaka)
- [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator)
- [highway - performance portable SIMD intrinsics](https://github.com/google/highway)
- [NVIDIA stdpar - GPU accelerated C++](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/)
- [Tuplex - Blazing fast python data science](https://tuplex.cs.brown.edu/)
- [Implicit SPMD Program Compiler (ISPC) - An open-source compiler for high-performance SIMD programming on the CPU and GPU](https://ispc.github.io/)
- [mpi4jax - zero-copy mpi for jax arrays](https://github.com/mpi4jax/mpi4jax) 
- [RS MPI - rust bindings for MPI](https://rsmpi.github.io/rsmpi/mpi/index.html)
- [async-rdma - A framework for writing RDMA applications with high-level abstraction and asynchronous APIs](https://github.com/datenlord/async-rdma)
- [joblib - data-flow programming for performance (python)](https://joblib.readthedocs.io/en/latest/why.html)
- [oneAPI - open, cross-industry, standards-based, unified, multiarchitecture, multi-vendor programming model](https://www.oneapi.io/)

#### Cluster Hardware Discovery Tools
 - [Likwid - provides all information about the supercomputer/cluster](https://github.com/RRZE-HPC/likwid)
 - [LIKWID.jl - julia wrapper for likwid](https://juliaperf.github.io/LIKWID.jl/dev/)
 - [cpuid](https://en.wikipedia.org/wiki/CPUID)
 - [cpuid instruction note](https://www.scss.tcd.ie/~jones/CS4021/processor-identification-cpuid-instruction-note.pdf)
 - [cpufetch](https://github.com/Dr-Noob/cpufetch)
 - [gpufetch](https://github.com/Dr-Noob/gpufetch)
 - [intel cpuinfo](https://www.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/command-reference/cpuinfo.html)
 - [openmpi hwloc](https://www.open-mpi.org/projects/hwloc/)
 
#### Cluster Management/Tools/Schedulers/Stacks

- [Flux framework](https://flux-framework.org/)
- [Bright Cluster Manager](https://www.brightcomputing.com/brightclustermanager)
- [E4S - The Extreme Scale HPC Scientific Stack](https://e4s-project.github.io/)
- [RADIUSS - Rapid Application Development via an Institutional Universal Software Stack](https://computing.llnl.gov/projects/radiuss)
- [OpenHPC](https://openhpc.community/)
- [Slurm](https://slurm.schedmd.com/overview.html)
- [SGE](http://star.mit.edu/cluster/docs/0.93.3/guides/sge.html)
- [Portable Batch System & OpenPBS](https://www.openpbs.org/)
- [Lustre Parallel File System](https://www.lustre.org/)
- [GPFS](https://en.wikipedia.org/wiki/GPFS)
- [Spack package manager for HPC/supercomputers](https://spack.io/)
- [Guix package manager for HPC/supercomputers](https://hpc.guix.info/)
- [Easybuild package manager for HPC/supercomputers](https://docs.easybuild.io/en/latest/)
- [Lmod](https://lmod.readthedocs.io/en/latest/)
- [Ruse](https://github.com/JanneM/Ruse)
- [xCat](https://xcat.org/)
- [Warewulf](https://warewulf.lbl.gov/)
- [Bluebanquise](https://github.com/bluebanquise/bluebanquise)
- [OpenXdMod](https://open.xdmod.org/7.5/index.html)
- [LSF](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=lsf-batch-jobs-tasks)
- [BeeGFS](http://beegfs.io/docs/whitepapers/Introduction_to_BeeGFS_by_ThinkParQ.pdf)
- [DeepOps - Nvidia GPU infrastructure and automation tools](https://github.com/NVIDIA/deepops)
- [fpsync - fast parallel data transfer using fpart and rsync](http://www.fpart.org/fpsync/)
- [moosefs - distributed file system](https://moosefs.com/)
- [rocks - open-source Linux cluster distribution](http://www.rocksclusters.org/)
- [sstack - a tool to install multiple software stacks, such as Spack, EasyBuild, and Conda](https://gitlab.com/nmsu_hpc/sstack)
- [DeepOps - Infrastructure automation tools for Kubernetes and Slurm clusters with NVIDIA GPUs](https://github.com/NVIDIA/deepops)

#### Development/Workflow/Monitoring Tools for HPC

- [Apptainer (formerly Singularity) - "the docker of HPC"](https://singularity.lbl.gov/)
- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/)
- [slurm docker cluster](https://github.com/giovtorres/slurm-docker-cluster)
- [Vaex - high performance dataframes in python](https://github.com/vaexio/vaex)
- [HTCondor](https://research.cs.wisc.edu/htcondor/)
- [grpc - high performance modern remote procedure call framework](https://grpc.io/)
- [Charliecloud](https://hpc.github.io/charliecloud/)
- [Jacamar-ci](https://gitlab.com/ecp-ci/jacamar-ci/-/blob/develop/README.md)
- [Prefect](https://www.prefect.io/)
- [Apache Airflow](https://airflow.apache.org/)
- [HPC Rocket - submit slurm jobs in CI](https://github.com/SvenMarcus/hpc-rocket)
- [Stui slurm dashboard for the terminal](https://github.com/mil-ad/stui)
- [Slurmvision slurm dashboard](https://github.com/Ruunyox/slurmvision)
- [genv - GPU Environment Management](https://github.com/run-ai/genv)
 
#### Debugging Tools for HPC

- [Summary of C/C++ debugging tools](http://pramodkumbhar.com/2018/06/summary-of-debugging-tools/)
- [ddt](https://www.arm.com/products/development-tools/server-and-hpc/forge/ddt)
- [totalview](https://totalview.io/)
- [marmot MPI checker](https://www.lrz.de/services/software/parallel/marmot/)
- [python debugging tools](https://wiki.python.org/moin/PythonDebuggingTools)
- [Differential Flamegraphs](https://www.brendangregg.com/blog/2014-11-09/differential-flame-graphs.html)

#### Performance/Benchmark Tools for HPC

- [Summary of code performance analysis tools](https://doku.lrz.de/display/PUBLIC/Performance+and+Code+Analysis+Tools+for+HPC)
- [papi](https://icl.utk.edu/papi/)
- [scalasca](https://www.scalasca.org/)
- [tau](https://www.cs.uoregon.edu/research/tau/home.php)
- [scalene](https://github.com/plasma-umass/scalene)
- [vampir](https://vampir.eu/)
- [kerncraft](https://github.com/RRZE-HPC/kerncraft)
- [NASA parallel benchmark suite](https://www.nas.nasa.gov/software/npb.html)
- [The Bandwidth Benchmark](https://github.com/RRZE-HPC/TheBandwidthBenchmark/)
- [Google benchmark](https://github.com/google/benchmark)
- [demonspawn](https://github.com/TACC/demonspawn)
- [HPL benchmark](https://www.netlib.org/benchmark/hpl/)
- [ngstress](https://github.com/ColinIanKing/stress-ng)
- [Ior](https://github.com/hpc/ior)
- [bytehound memory profiler](https://github.com/koute/bytehound)
- [Flamegraphs](https://www.brendangregg.com/flamegraphs.html)
- [fio](https://linux.die.net/man/1/fio)
- [IBM Spectrum Scale Key Performance Indicators (KPI)](https://github.com/IBM/SpectrumScale_NETWORK_READINESS)

#### IO/Visualization Tools for HPC
 - [the yt project](https://yt-project.org/)
 - [paraview](https://yt-project.org/)
 - [visit](https://wci.llnl.gov/simulation/computer-codes/visit)
 - [vedo](https://vedo.embl.es/)
 - [Amira](https://www.thermofisher.com/ca/en/home/electron-microscopy/products/software-em-3d-vis/amira-software.html)
 - [Scientific Visualization Wiki](https://en.wikipedia.org/wiki/Scientific_visualization)
 - [ADIOS2](https://github.com/ornladios/ADIOS2)

#### General Purpose Scientific Computing Libraries for HPC
 - [petsc](https://petsc.org/release/)
 - [ginkgo](https://ginkgo-project.github.io/)
 - [GSL](https://www.gnu.org/software/gsl/)
 - [Scalapack](https://netlib.org/scalapack/)
 - [rapids.ai - collection of libraries for executing end-to-end data science pipelines completely in the GPU](rapids.ai)
 
#### Misc.
 - [mimalloc memory allocator](https://github.com/microsoft/mimalloc)
 - [jemalloc memory allocator](https://github.com/jemalloc/jemalloc)
 - [tcmalloc memory allocator](https://github.com/google/tcmalloc)
 - [Horde memory allocator](https://github.com/emeryberger/Hoard)

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

### CPU
- [Wikichip](https://en.wikichip.org/wiki/WikiChip)
- [Microarchitecture of Intel/AMD CPUs](https://www.agner.org/optimize/microarchitecture.pdf)
- [Apple M1](https://en.wikipedia.org/wiki/Apple_M1)
- [Apple M2](https://en.wikipedia.org/wiki/Apple_M2)
- [Apple M2 Teardown](https://www.ifixit.com/News/62674/m2-macbook-air-teardown-apple-forgot-the-heatsink)
- [Apply M1/M2 AMX](https://github.com/corsix/amx)
- [Comparison of Intel processors](https://en.wikipedia.org/wiki/Comparison_of_Intel_processors)
- [Comparison of Apple processors](https://en.wikipedia.org/wiki/Apple-designed_processors)
- [Comparison of AMD architectures](https://en.wikipedia.org/wiki/Table_of_AMD_processors)

### GPU

- [Gpu Architecture Analysis](https://graphicscodex.courses.nvidia.com/app.html?page=_rn_parallel)
- [A trip through the Graphics Pipeline](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)
- [A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/)
- [Gentle Intro to GPU Inner Workings](https://vksegfault.github.io/posts/gentle-intro-gpu-inner-workings/)
- [AMD Instinct GPUs](https://en.wikipedia.org/wiki/AMD_Instinct_accelerators)
- [AMD GPU List Wiki](https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units)
- [Comparison of CUDA architectures](https://en.wikipedia.org/wiki/CUDA)
- [Tales of the M1 GPU](https://asahilinux.org/2022/11/tales-of-the-m1-gpu/)

### TPU/Tensor Cores

- [Google TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)

### Many integrated core processor (MIC)

- [Xeon Phi](https://en.wikipedia.org/wiki/Xeon_Phi)

### Cloud

#### Vendors

- [AWS HPC](https://aws.amazon.com/hpc/)
- [Azure HPC](https://azure.microsoft.com/en-us/solutions/high-performance-computing/#intro)
- [rescale](https://rescale.com/)
- [vast.ai](https://vast.ai/)
- [vultr - cheap bare metal CPU, GPU, DGX servers](vultr.com)
- [hetzner - cheap servers incl. 80-core ARM](https://www.hetzner.com/)
- [Ampere ARM cloud-native processors](https://amperecomputing.com/)
- [Scaleway](https://www.scaleway.com/en/)

#### Articles/Papers
- [The use of Microsoft Azure for high performance cloud computing – A case study](https://www.diva-portal.org/smash/get/diva2:1704798/FULLTEXT01.pdf)
- [AWS Cluster in the cloud](https://cluster-in-the-cloud.readthedocs.io/en/latest/aws-infrastructure.html)
- [AWS Parallel Cluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorials-running-your-first-job-on-version-3.html)

### Custom/FPGA/ASIC/APU

- [OpenPiton](http://parallel.princeton.edu/openpiton/)
- [Parallela](https://www.parallella.org/)
- [AMD APU](https://en.wikipedia.org/wiki/AMD_Accelerated_Processing_Unit)

### Certification

- [Intel Cluster Ready](https://en.wikipedia.org/wiki/Intel_Cluster_Ready)

### Other/Wikis

- [Supercomputer](https://en.wikipedia.org/wiki/Supercomputer)
- [Supercomputer architecture](https://en.wikipedia.org/wiki/Supercomputer_architecture)
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

## People

 - [Jack Dongarra - 2021 Turing Award - LINPACK, BLAS, LAPACK, MPI](https://www.nature.com/articles/s43588-022-00245-w)
 - [Bill Gropp - 2010 IEEE TCSC Medal for Excellence in Scalable Computing](https://en.wikipedia.org/wiki/Bill_Gropp)
 - [David Bader - built the first Linux supercomputer](https://en.wikipedia.org/wiki/David_Bader_(computer_scientist))
 - [Thomas Sterling - Inventor of Beowulf cluster, ParalleX/HPX](https://en.wikipedia.org/wiki/Thomas_Sterling_(computing))
 - [Seymour Cray - Inventor of the Cray Supercomputer](https://en.wikipedia.org/wiki/Seymour_Cray)
 - [Larry Smarr - HPC Application Pioneer](https://en.wikipedia.org/wiki/Larry_Smarr)
  
## Resources

#### Books/Manuals
- [HPC Books by Victor Eijkhout](https://theartofhpc.com/)
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

#### Courses
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

#### Tutorials/Guides/Articles
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

#### Review Papers/Articles
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

#### News
- [InsideHPC](https://insidehpc.com/)
- [HPCWire](https://www.hpcwire.com/)
- [NextPlatform](https://www.nextplatform.com)
- [Datacenter Dynamics](https://www.datacenterdynamics.com/en/)
- [Admin Magazine HPC](https://www.admin-magazine.com/HPC/News)
- [Toms hardware](https://www.tomshardware.com/)
- [Tech Radar](https://www.techradar.com/)

#### Podcasts
- [This week in HPC](https://soundcloud.com/this-week-in-hpc)
- [Preparing Applications for Aurora in the Exascale Era](https://connectedsocialmedia.com/20114/preparing-applications-for-aurora-in-the-exascale-era/)

#### Youtube Videos/Courses
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

#### Presentation Slides
- [Task based Parallelism and why it's awesome](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Workshops/Conferences/CSAM-2015/Programme/lecture7a_gonnet-pdf.pdf?__blob=publicationFile) - Pedro Gonnet
- [Tuning Slurm Scheduling for Optimal Responsiveness and Utilization](https://slurm.schedmd.com/SUG14/sched_tutorial.pdf)
- [Parallel Programming Models Overview (2020)](https://www.researchgate.net/publication/348187154_Parallel_programming_models_overview_2020)
- [Comparative Analysis of Kokkos and Sycl (Jeff Hammond)](https://www.iwocl.org/wp-content/uploads/iwocl-2019-dhpcc-jeff-hammond-a-comparitive-analysis-of-kokkos-and-sycl.pdf) 
- [Hybrid OpenMP/MPI Programming](https://www.nersc.gov/assets/Uploads/NUG2013hybridMPIOpenMP2.pdf)
- [Designs, Lessons and Advice from Building Large Distributed Systems - Jeff Dean (Google)](http://www.cs.cornell.edu/projects/ladis2009/talks/dean-keynote-ladis2009.pdf)

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
 
#### Membership Clubs
 - [Association for Computing Machinery](acm.org)
 - [ETP4HPC](https://www.etp4hpc.eu/)
 
#### Blogs
 - [1024 Cores](http://www.1024cores.net/) - Dmitry Vyukov 
 - [The Black Art of Concurrency](https://www.internalpointers.com/post-group/black-art-concurrency) - Internal Pointers
 - [Cluster Monkey](https://www.clustermonkey.net/)
 - [Johnathon Dursi](https://www.dursi.ca/)
 - [Arm Vendor HPC blog](https://community.arm.com/developer/tools-software/hpc/b/hpc-blog)
 - [HPC Notes](https://www.hpcnotes.com/)
 - [Brendan Gregg Performance Blog](https://www.brendangregg.com/blog/index.html)

#### Journals
 - [IEEE Transactions on Parallel and Distributed Systems (TPDS)](https://www.computer.org/csdl/journal/td) 
 - [Journal of Parallel and Distributed Computing](https://www.journals.elsevier.com/journal-of-parallel-and-distributed-computing)
  
#### Conferences

 - [ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP)](https://ppopp19.sigplan.org/home)
 - [ACM Symposium on Parallel Algorithms and Architectures (SPAA)](https://spaa.acm.org/)
 - [ACM/IEEE International Conference for High-performance Computing, Networking, Storage, and Analysis (SC)](https://sc19.supercomputing.org/)
 - [IEEE International Parallel and Distributed Processing Symposium (IPDPS)](http://www.ipdps.org/)
 - [International Conference on Parallel Processing (ICPP)](https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/)

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

#### Building Clusters
- [Build a cluster under 50k](https://www.reddit.com/r/HPC/comments/srssrt/build_a_minicluster_under_50000/)
- [Build a Beowulf cluster](https://github.com/darshanmandge/Cluster) 
- [Build a Raspberry Pi Cluster](https://www.raspberrypi.com/tutorials/cluster-raspberry-pi-tutorial/)
- [Puget Systems](https://www.pugetsystems.com/)
- [Lambda Systems](https://lambdalabs.com/)
- [Titan computers](https://www.titancomputers.com)
- [Temple course on building/maintaining a cluster](https://www.hpc.temple.edu/mhpc/2021/hpc-technology/index.html)
- [Detailed reddit discussion on setting up a small cluster](https://www.reddit.com/r/HPC/comments/xeipt7/setting_up_a_small_hpc_cluster/)
- [Tiny titan - build a really cool pi supercomputer](https://tinytitan.github.io/)

#### Misc. Papers/Articles
- [Advanced Parallel Programming in C++](https://www.diehlpk.de/assets/modern_cpp.pdf)
- [Tools for scientific computing](https://arxiv.org/pdf/2108.13053.pdf)
- [Quantum Computing for High Performance Computing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9537178)
- [Benchmarking data science: Twelve ways to lie with statistics and performance on parallel computers.](http://ww.unixer.de/publications/img/hoefler-12-ways-data-science-preprint.pdf)
- [Establishing the IO500 Benchmark](https://www.vi4io.org/_media/io500/about/io500-establishing.pdf)

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
  
#### Misc. Theses
   - [Rust programming language in the high-performance computing environment](https://www.research-collection.ethz.ch/handle/20.500.11850/474922)
  
#### Other Curated Lists
   - [Parallel Computing Guide](https://github.com/mikeroyal/Parallel-Computing-Guide)
   - [Awesome Parallel Computing](https://github.com/taskflow/awesome-parallel-computing)
   

## Acknowledgements

This repo started from the great curated list https://github.com/taskflow/awesome-parallel-computing


