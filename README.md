# Awesome High Performance Computing Resources

A curated list of awesome high performance computing resources. 
Please feel free to update this page through submitting pull requests.

## Table of Contents

 - [General Info](#general-info)
 - [Software](#software)
 - [Hardware](#hardware) 
 - [Resources](#resources)
 - [Acknowledgements](#acknowledgements )

## General Info

### A Few Upcoming Supercomputers 
 - [Frontier](https://www.amd.com/en/products/exascale-era) - 2021, AMD-based, ~1.5 exaflops
 - [El Capitan](https://www.amd.com/en/products/exascale-era) - 2023, AMD-based, ~1.5 exaflops
 - [Aurora](https://en.wikipedia.org/wiki/Aurora_(supercomputer)) - 2021, Intel-based, ~1 exaflops
 - [Tianhe-3](https://www.nextplatform.com/2019/05/02/china-fleshes-out-exascale-design-for-tianhe-3/) - 2021, ~700 Petaflop (Linpack500)

### Current List of Top 500 Supercomputers
 - https://www.top500.org/lists/top500/2021/11/

## Software

#### Popular HPC Programming Libraries/APIs/Tools

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
- [rapids.ai - collection of libraries for executing end-to-end data science pipelines completely in the GPU](rapids.ai)
- [HPC-X - Nvidia implementation of MPI](https://developer.nvidia.com/networking/hpc-x)
- [MPAVICH - Implementation of MPI](https://mvapich.cse.ohio-state.edu/)
- [mpi4py - python bindings for MPI](https://mpi4py.readthedocs.io/en/stable/)
- [UCX - optimized production proven-communication framework](https://github.com/openucx/ucx#using-ucx)
- [Horovod - distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet](https://github.com/horovod/horovod)
- [Taichi - parallel programming language for high-performance numerical computations (embedded in Python with JIT support)](https://github.com/taichi-dev/taichi)
- [MAGMA - next generation linear algebra (LA) GPU accelerated libraries](https://developer.nvidia.com/magma)
- [NVIDIA cuNumeric - GPU drop-in for numpy](https://developer.nvidia.com/cunumeric)

#### Cluster Hardware Discovery Tools
 - [Likwid - provides all information about the supercomputer/cluster](https://github.com/RRZE-HPC/likwid)
 - [LIKWID.jl - julia wrapper for likwid](https://juliaperf.github.io/LIKWID.jl/dev/)
 - [cpuid](https://en.wikipedia.org/wiki/CPUID)
 - [cpuid instruction note](https://www.scss.tcd.ie/~jones/CS4021/processor-identification-cpuid-instruction-note.pdf)
 - [cpufetch](https://github.com/Dr-Noob/cpufetch)
 - [gpufetch](https://github.com/Dr-Noob/gpufetch)
 
#### Cluster Management/Tools/Stacks

- [E4S - The Extreme Scale HPC Scientific Stack](https://e4s-project.github.io/)
- [RADIUSS - Rapid Application Development via an Institutional Universal Software Stack](https://computing.llnl.gov/projects/radiuss)
- [OpenHPC](https://openhpc.community/)
- [Slurm](https://slurm.schedmd.com/overview.html)
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

#### Development/Workflow Tools for HPC

- [Singularity](https://singularity.lbl.gov/)
- [Vaex - high performance dataframes in python](https://github.com/vaexio/vaex)
- [HTCondor](https://research.cs.wisc.edu/htcondor/)
- [grpc - high performance modern remote procedure call framework](https://grpc.io/)
- [Charliecloud](https://hpc.github.io/charliecloud/)
- [Jacamar-ci](https://gitlab.com/ecp-ci/jacamar-ci/-/blob/develop/README.md)
- [Prefect](https://www.prefect.io/)
- [Apache Airflow](https://airflow.apache.org/)
- [HPC Rocket - submit slurm jobs in CI](https://github.com/SvenMarcus/hpc-rocket)

#### Debugging Tools for HPC

- [Summary of C/C++ debugging tools](http://pramodkumbhar.com/2018/06/summary-of-debugging-tools/)
- [ddt](https://www.arm.com/products/development-tools/server-and-hpc/forge/ddt)
- [totalview](https://totalview.io/)
- [marmot MPI checker](https://www.lrz.de/services/software/parallel/marmot/)
- [python debugging tools](https://wiki.python.org/moin/PythonDebuggingTools)

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

#### Wikis

- [Comparison of cluster software](https://en.wikipedia.org/wiki/Comparison_of_cluster_software)
- [List of cluster management software](https://en.wikipedia.org/wiki/List_of_cluster_management_software)

## Hardware

### Interconnects

- [Network topologies](https://www.hpcwire.com/2019/07/15/super-connecting-the-supercomputers-innovations-through-network-topologies/)
- [Battle of the infinibands - Omnipath vs Infiniband](https://www.nextplatform.com/2017/11/29/the-battle-of-the-infinibands/)
- [Mellanox infiniband cluster config](https://www.mellanox.com/clusterconfig/)
- [RoCE - RDMA Over Converged Ethernet](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet)

### CPU
- [Microarchitecture of Intel/AMD CPUs](https://www.agner.org/optimize/microarchitecture.pdf)

### GPU

- [Gpu Architecture Analysis](https://graphicscodex.courses.nvidia.com/app.html?page=_rn_parallel)
- [A trip through the Graphics Pipeline](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)
- [A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/)

### TPU/Tensor Cores

- [Google TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)

### Cloud

- [Cluster in the cloud](https://cluster-in-the-cloud.readthedocs.io/en/latest/aws-infrastructure.html)
- [AWS Parallel Cluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorials-running-your-first-job-on-version-3.html)

### Wikis

- [Supercomputer](https://en.wikipedia.org/wiki/Supercomputer)
- [Supercomputer architecture](https://en.wikipedia.org/wiki/Supercomputer_architecture)
- [Computer cluster](https://en.wikipedia.org/wiki/Computer_cluster)
- [Infinband](https://en.wikipedia.org/wiki/InfiniBand#:~:text=InfiniBand%20(IB)%20is%20a%20computer,both%20among%20and%20within%20computers.)
- [Comparison of Intel processors](https://en.wikipedia.org/wiki/Comparison_of_Intel_processors)
- [Comparison of Apple processors](https://en.wikipedia.org/wiki/Apple-designed_processors)
- [Comparison of AMD architectures](https://en.wikipedia.org/wiki/Table_of_AMD_processors)
- [Comparison of CUDA architectures](https://en.wikipedia.org/wiki/CUDA)
- [Cache](https://en.wikipedia.org/wiki/Cache_(computing))
- [Google TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)

## Resources

#### Books/Manuals
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

#### Courses
- [Berkeley: Applications of Parallel Computers](https://sites.google.com/lbl.gov/cs267-spr2019/) - Detailed course on HPC
- [CS6290 High-performance Computer Architecture](https://www.udacity.com/course/high-performance-computer-architecture--ud007) - Milos Prvulovic and Catherine Gamboa at George Tech
- [Udacity High Performance Computing](https://www.youtube.com/playlist?list=PLAwxTw4SYaPk8NaXIiFQXWK6VPnrtMRXC)
- [Parallel Numerical Algorithms](https://solomonik.cs.illinois.edu/teaching/cs554/index.html)
- [Vanderbilt - Intro to HPC](https://github.com/vanderbiltscl/SC3260_HPC)
- [Illinois - Intro to HPC](https://andreask.cs.illinois.edu/Teaching/HPCFall2012/) - Creator of PyCuda
- [Archer Courses](http://www.archer.ac.uk/training/past_courses.php)
- [TACC tutorials](https://portal.tacc.utexas.edu/tutorials)
- [Livermore training materials](https://hpc.llnl.gov/training/tutorials)
- [Xsede training materials](https://www.hpc-training.org/xsede/moodle/)
- [Parallel Computation Math](https://www.cct.lsu.edu/~pdiehl/teaching/2021/4997/)
- [Introduction to High-Performance and Parallel Computing - Coursera](https://www.coursera.org/learn/introduction-high-performance-computing)

#### Tutorials/Guides
- [MpiTutorial](mpitutorial.com) - A fantastic mpi tutorial
- [Beginners Guide to HPC](http://www.shodor.org/petascale/materials/UPModules/beginnersGuideHPC/)
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

#### Review Papers
- [The Landscape of Exascale Research: A Data-Driven Literature Analysis (2020)](https://dl.acm.org/doi/pdf/10.1145/3372390)
- [The Landscape of Parallel Computing Research: A View from Berkeley](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf)
- [Extreme Heterogeneity 2018: Productive Computational Science in the Era of Extreme Heterogeneity](references/2018-Extreme-Heterogeneity-DoE.pdf)
- [Programming for Exascale Computers - Will Gropp, Marc Snir](https://snir.cs.illinois.edu/listed/J55.pdf)
- [On the Memory Underutilization: Exploring Disaggregated Memory on HPC Systems (2020)](https://www.mcs.anl.gov/research/projects/argo/publications/2020-sbacpad-peng.pdf)
- [Advances in Parallel & Distributed Processing, and Applications (conference proceedings)](https://link.springer.com/book/10.1007/978-3-030-69984-0)
- [Designing Heterogeneous Systems: Large Scale Architectural Exploration Via Simulation](https://ieeexplore.ieee.org/abstract/document/9651152)

#### News
- [InsideHPC](https://insidehpc.com/)
- [HPCWire](https://www.hpcwire.com/)
- [NextPlatform](https://www.nextplatform.com)
- [Datacenter Dynamics](https://www.datacenterdynamics.com/en/)
- [Admin Magazine HPC](https://www.admin-magazine.com/HPC/News)

#### Podcasts
- [This week in HPC](https://soundcloud.com/this-week-in-hpc)

#### Youtube
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

#### Presentation Slides
- [Task based Parallelism and why it's awesome](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Workshops/Conferences/CSAM-2015/Programme/lecture7a_gonnet-pdf.pdf?__blob=publicationFile) - Pedro Gonnet
- [Tuning Slurm Scheduling for Optimal Responsiveness and Utilization](https://slurm.schedmd.com/SUG14/sched_tutorial.pdf)
- [Parallel Programming Models Overview (2020)](https://www.researchgate.net/publication/348187154_Parallel_programming_models_overview_2020)
- [Comparative Analysis of Kokkos and Sycl (Jeff Hammond)](https://www.iwocl.org/wp-content/uploads/iwocl-2019-dhpcc-jeff-hammond-a-comparitive-analysis-of-kokkos-and-sycl.pdf)

#### Forums
 - [r/hpc](https://www.reddit.com/r/HPC/)
 - [r/homelab](https://www.reddit.com/r/homelab/)

#### Careers
 - [HPC University Careers search](http://hpcuniversity.org/careers/)
 - [HPC wire career site](https://careers.hpcwire.com/)

#### Blogs
 - [1024 Cores](http://www.1024cores.net/) - Dmitry Vyukov 
 - [The Black Art of Concurrency](https://www.internalpointers.com/post-group/black-art-concurrency) - Internal Pointers
 - [Cluster Monkey](https://www.clustermonkey.net/)
 - [Johnathon Dursi](https://www.dursi.ca/)
 - [Arm Vendor HPC blog](https://community.arm.com/developer/tools-software/hpc/b/hpc-blog)
 - [HPC Notes](https://www.hpcnotes.com/)

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

#### Consulting
- [Redline Performance](https://redlineperf.com/)
- [R systems](http://rsystemsinc.com/)
- [Advanced Clustering](https://www.advancedclustering.com/)

#### HPC Interview Preparation
  - [Reddit Entry Level HPC interview help](https://www.reddit.com/r/HPC/comments/nhpdfb/entrylevel_hpc_job_interview/)

#### Organizations
  - [Prace](https://prace-ri.eu/)
  - [Xsede](https://www.xsede.org/)
  - [Compute Canada](https://www.computecanada.ca/)

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

#### Misc. Papers
- [Advanced Parallel Programming in C++](https://www.diehlpk.de/assets/modern_cpp.pdf)
- [Tools for scientific computing](https://arxiv.org/pdf/2108.13053.pdf)
- [Quantum Computing for High Performance Computing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9537178)

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

#### Misc.
  - [Pocket HPC Survival Guide](https://tin6150.github.io/psg/lsf.html)
  - [HPC Summer school](https://www.ihpcss.org/)
  - [Overview of all linear algebra packages](http://www.netlib.org/utk/people/JackDongarra/la-sw.html)
  - [Latency numbers](http://norvig.com/21-days.html#answers)
  - [Nvidia HPC benchmarks](https://ngc.nvidia.com/catalog/containers/nvidia:hpc-benchmarks)
  - [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#)
  - [AWS Cloud calculator](https://calculator.aws/)
  - [Quickly benchmark C++ functions](https://quick-bench.com/)

## Acknowledgements

This repo started from the great curated list https://github.com/taskflow/awesome-parallel-computing


