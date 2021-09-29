# Awesome High Performance Computing Resources

A curated list of awesome high performance computing resources. 
Please feel free to update this page through submitting pull requests.

## Table of Contents

 - [General Info](#general-info)
 - [Software](#software)
 - [Hardware](#hardware) 
 - [Learning Resources](#learning-resources)
 - [Acknowledgements](#acknowledgements )

## General Info

### A Few Upcoming Supercomputers 
 - [Frontier](https://www.amd.com/en/products/exascale-era) - 2021, AMD-based, ~1.5 exaflops
 - [El Capitan](https://www.amd.com/en/products/exascale-era) - 2023, AMD-based, ~1.5 exaflops
 - [Aurora](https://en.wikipedia.org/wiki/Aurora_(supercomputer)) - 2021, Intel-based, ~1 exaflops
 - [Tianhe-3](https://www.nextplatform.com/2019/05/02/china-fleshes-out-exascale-design-for-tianhe-3/) - 2021, ~700 Petaflop (Linpack500)

### Current List of Top 500 Supercomputers
 - https://www.top500.org/lists/top500/2021/06/

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

#### Cluster Management/Tools

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

#### Development/Workflow Tools for HPC

- [Singularity](https://singularity.lbl.gov/)
- [Vaex - high performance dataframes in python](https://github.com/vaexio/vaex)
- [HTCondor](https://research.cs.wisc.edu/htcondor/)
- [grpc - high performance modern remote procedure call framework](https://grpc.io/)
- [Charliecloud](https://hpc.github.io/charliecloud/)

#### Debugging Tools for HPC

- [Summary of debugging tools](http://pramodkumbhar.com/2018/06/summary-of-debugging-tools/)
- [ddt](https://www.arm.com/products/development-tools/server-and-hpc/forge/ddt)
- [totalview](https://totalview.io/)
- [marmot MPI checker](https://www.lrz.de/services/software/parallel/marmot/)

#### Performance Tools for HPC

- [Summary of code performance analysis tools](https://doku.lrz.de/display/PUBLIC/Performance+and+Code+Analysis+Tools+for+HPC)
- [papi](https://icl.utk.edu/papi/)
- [scalasca](https://www.scalasca.org/)

## Hardware

### Interconnects

- [Network topologies](https://www.hpcwire.com/2019/07/15/super-connecting-the-supercomputers-innovations-through-network-topologies/)
- [Battle of the infinibands - Omnipath vs Infiniband](https://www.nextplatform.com/2017/11/29/the-battle-of-the-infinibands/)
- [Mellanox infiniband cluster config](https://www.mellanox.com/clusterconfig/)

## Learning Resources

#### Books
- [Introduction to High Performance Scientific Computing](https://web.corral.tacc.utexas.edu/CompEdu/pdf/stc/EijkhoutIntroToHPC.pdf) - Victor Eijkhout 2021
- [Parallel Programming for Science and Engineering](https://web.corral.tacc.utexas.edu/CompEdu/pdf/pcse/EijkhoutParallelProgramming.pdf) - Victor EIjkhout 2021
- [Data Parallel C++ Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL](https://www.apress.com/gp/book/9781484255735)
- [C++ Concurrency in Action: Practical Multithreading](https://www.manning.com/books/c-plus-plus-concurrency-in-action) - Anthony Williams 2012
- [The Art of Multiprocessor Programming](https://www.amazon.com/Art-Multiprocessor-Programming-Revised-Reprint/dp/0123973376/ref=sr_1_1?ie=UTF8&qid=1438003865&sr=8-1&keywords=maurice+herlihy) - Maurice Herlihy 2012
- [Parallel Computing: Theory and Practice](http://www.cs.cmu.edu/afs/cs/academic/class/15210-f15/www/tapp.html#ch:work-stealing) - Umut A. Acar 2016
- [Introduction to Parallel Computing](https://www.amazon.ca/Introduction-Parallel-Computing-Zbigniew-Czech/dp/1107174392/ref=sr_1_7?dchild=1&keywords=parallel+computing&qid=1625711415&sr=8-7) - Zbigniew J. Czech

#### Tutorials
- [MpiTutorial](mpitutorial.com) - A fantastic mpi tutorial
- [Beginners Guide to HPC](http://www.shodor.org/petascale/materials/UPModules/beginnersGuideHPC/)
- [Parallel Computing Training Tutorials](https://hpc.llnl.gov/training/tutorials) - Lawrence Livermore National Laboratory
- [Foundations of Multithreaded, Parallel, and Distributed Programming](https://www.amazon.com/Foundations-Multithreaded-Parallel-Distributed-Programming/dp/B00F4I7HM2/ref=sr_1_2?dchild=1&keywords=Gregory+R.+Andrews+Distributed+Programming&qid=1625766665&s=books&sr=1-2)
- [Building pipelines using slurm dependencies](https://hpc.nih.gov/docs/job_dependencies.html)
- [Writing slurm scripts in python,r and bash](https://vsoch.github.io/lessons/sherlock-jobs/)
- [Xsede new user tutorials](https://portal.xsede.org/online-training)
- [Supercomputing in plain english](http://www.oscer.ou.edu/education.php)

#### Review Papers
- [The Landscape of Exascale Research: A Data-Driven Literature Analysis (2020)](https://dl.acm.org/doi/pdf/10.1145/3372390)
- [The Landscape of Parallel Computing Research: A View from Berkeley](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf)
- [Extreme Heterogeneity 2018: Productive Computational Science in the Era of Extreme Heterogeneity](references/2018-Extreme-Heterogeneity-DoE.pdf)
- [Programming for Exascale Computers - Will Gropp, Marc Snir](https://snir.cs.illinois.edu/listed/J55.pdf)

#### Misc Papers
- [Tools for scientific computing](https://arxiv.org/pdf/2108.13053.pdf)
- [Quantum Computing for High Performance Computing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9537178)

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

#### News
- [InsideHPC](https://insidehpc.com/)
- [HPCWire](https://www.hpcwire.com/)
- [NextPlatform](https://www.nextplatform.com)
- [Datacenter Dynamics](https://www.datacenterdynamics.com/en/)
- [Admin Magazine HPC](https://www.admin-magazine.com/HPC/News)

#### Podcasts
- [This week in HPC](https://soundcloud.com/this-week-in-hpc)

#### Consulting
- [Redline Performance](https://redlineperf.com/)
- [R systems](http://rsystemsinc.com/)
- [Advanced Clustering](https://www.advancedclustering.com/)

#### Wikis
- [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law)
- [HPC Wiki](https://hpc-wiki.info/hpc/HPC_Wiki)
- [Supercomputer](https://en.wikipedia.org/wiki/Supercomputer)
- [Supercomputer architecture](https://en.wikipedia.org/wiki/Supercomputer_architecture)
- [Computer cluster](https://en.wikipedia.org/wiki/Computer_cluster)
- [Comparison of cluster software](https://en.wikipedia.org/wiki/Comparison_of_cluster_software)
- [List of cluster management software](https://en.wikipedia.org/wiki/List_of_cluster_management_software)
- [Infinband](https://en.wikipedia.org/wiki/InfiniBand#:~:text=InfiniBand%20(IB)%20is%20a%20computer,both%20among%20and%20within%20computers.)
- [Comparison of Intel processors](https://en.wikipedia.org/wiki/Comparison_of_Intel_processors)
- [Comparison of Apple processors](https://en.wikipedia.org/wiki/Apple-designed_processors)
- [Comparison of AMD architectures](https://en.wikipedia.org/wiki/Table_of_AMD_processors)
- [Comparison of CUDA architectures](https://en.wikipedia.org/wiki/CUDA)
- [FLOPS](https://en.wikipedia.org/wiki/FLOPS)
- [Computational complexity of math operations](https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations)
- [Many Task Computing](https://en.wikipedia.org/wiki/Many-task_computing)
- [High Throughput Computing](https://en.wikipedia.org/wiki/High-throughput_computing)
- [Parallel Virtual Machine](https://en.wikipedia.org/wiki/Parallel_Virtual_Machine)
- [OSI Model](https://en.wikipedia.org/wiki/OSI_model)
- [Workflow management](https://en.wikipedia.org/wiki/Scientific_workflow_system)
- [Compute Canada Documentation](https://docs.computecanada.ca/wiki/Compute_Canada_Documentation)
- [Network Interface Controller (NIC)](https://en.wikipedia.org/wiki/Network_interface_controller)

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

#### Presentation Slides
- [Task based Parallelism and why it's awesome](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Workshops/Conferences/CSAM-2015/Programme/lecture7a_gonnet-pdf.pdf?__blob=publicationFile) - Pedro Gonnet

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

#### Misc. Repos
  - [Build a Beowulf cluster](https://github.com/darshanmandge/Cluster)
  - [libsc - Supercomputing library](https://github.com/cburstedde/libsc)
  - [xbyak jit assembler](https://github.com/herumi/xbyak)
  - [cpufetch - pretty cpu info fetcher](https://github.com/Dr-Noob/cpufetch)
  - [RRZE-HPC](https://github.com/RRZE-HPC)
  - [Argonne Github](https://github.com/Argonne-National-Laboratory)
  - [Oak Ridge National Lab Github](https://github.com/ORNL)
  - [HPCInfo by Jeff Hammond](https://github.com/jeffhammond/HPCInfo)

#### Other
  - [Nice notes](https://caiorss.github.io/C-Cpp-Notes/Libraries.html)
  - [Pocket HPC Survival Guide](https://tin6150.github.io/psg/lsf.html)
  - [HPC Summer school](https://www.ihpcss.org/)
  - [Overview of all linear algebra packages](http://www.netlib.org/utk/people/JackDongarra/la-sw.html)
  - [Latency numbers](http://norvig.com/21-days.html#answers)
  - [Nvidia HPC benchmarks](https://ngc.nvidia.com/catalog/containers/nvidia:hpc-benchmarks)

#### HPC Interview Preparation
  - [Reddit Entry Level HPC interview help](https://www.reddit.com/r/HPC/comments/nhpdfb/entrylevel_hpc_job_interview/)

#### Organizations
  - [Prace](https://prace-ri.eu/)
  - [Xsede](https://www.xsede.org/)
  - [Compute Canada](https://www.computecanada.ca/)

## Acknowledgements

This repo started from the great curated list https://github.com/taskflow/awesome-parallel-computing


