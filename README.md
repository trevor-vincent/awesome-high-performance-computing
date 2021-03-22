# Awesome High Performance Computing Resources

A curated list of awesome high performance computing resources. 
Please feel free to update this page through submitting pull requests.

## Table of Contents

 - [General Info](#general-info)
 - [Software](#software)
 - [Hardware](#hardware) 
 - [Presentations](#presentations)
 - [Learning Resources](#learning-resources)
 - [Links](#links)
 - [Acknowledgements](#acknowledgements )

## General Info

### Supercomputers coming in the future
 - [Frontier](https://www.amd.com/en/products/exascale-era) - 2021, AMD-based, ~1.5 exaflops
 - [El Capitan](https://www.amd.com/en/products/exascale-era) - 2023, AMD-based, ~1.5 exaflops
 - [Aurora](https://en.wikipedia.org/wiki/Aurora_(supercomputer)) - 2021, Intel-based, ~1 exaflops

### Current List of Supercomputers
 - https://www.top500.org/lists/top500/2020/11/

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
- [CUDA: High performance NVIDIA GPU acceleration](https://developer.nvidia.com/cuda-toolkit)
- [OpenACC: "OpenMP for GPUs"](https://www.openacc.org/)
- [numba: Numba is an open source JIT compiler that translates a subset of Python into fast machine code.](https://numba.pydata.org/)
- [dask: Dask provides advanced parallelism for analytics, enabling performance at scale for the tools you love](https://dask.org)

#### Cluster Management/Tools

- [Slurm](https://slurm.schedmd.com/overview.html)
- [Portable Batch System & OpenPBS](https://www.openpbs.org/)
- [Lustre Parallel File System](https://www.lustre.org/)
- [GPFS](https://en.wikipedia.org/wiki/GPFS)
- [Spack package manager for supercomputers](https://spack.io/)
- [Guix package manager for supercomputers](https://hpc.guix.info/)
- [Lmod](https://lmod.readthedocs.io/en/latest/)

#### Development Tools for HPC

- [Singularity](https://singularity.lbl.gov/)

#### Workflow managment for HPC

- [HTCondor](https://research.cs.wisc.edu/htcondor/)

#### Debugging Tools for HPC

- [ddt](https://www.arm.com/products/development-tools/server-and-hpc/forge/ddt)
- [totalview](https://totalview.io/)

## Hardware

### Interconnects

- [Network topologies](https://www.hpcwire.com/2019/07/15/super-connecting-the-supercomputers-innovations-through-network-topologies/)
- [Battle of the infinibands - Omnipath vs Infiniband](https://www.nextplatform.com/2017/11/29/the-battle-of-the-infinibands/)

## Presentations

#### Generic Parallel Computing Topics

- [Task based Parallelism and why it's awesome](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Workshops/Conferences/CSAM-2015/Programme/lecture7a_gonnet-pdf.pdf?__blob=publicationFile) - Pedro Gonnet
- [Concurrency in C++20 and Beyond](https://www.youtube.com/watch?v=jozHW_B3D4U) - A. Williams
- [Is Parallel Programming still Hard?](https://www.youtube.com/watch?v=YM8Xy6oKVQg) - P. McKenney, M. Michael, and M. Wong at CppCon 2017
- [The Speed of Concurrency: Is Lock-free Faster?](https://www.youtube.com/watch?v=9hJkWwHDDxs) - Fedor G Pikus in CppCon 2016

#### Scheduling in Parallel Processing

- [Expressing Parallelism in C++ with Threading Building Blocks](https://www.youtube.com/watch?v=9Otq_fcUnPE) - Mike Voss at Intel Webinar 2018
- [A Work-stealing Runtime for Rust](https://www.youtube.com/watch?v=4DQakkJ8XLI) - Aaron Todd in Air Mozilla 2017

#### Memory Model

- [C++11/14/17 atomics and memory model: Before the story consumes you](https://www.youtube.com/watch?v=DS2m7T6NKZQ) - Michael Wong in CppCon 2015
- [The C++ Memory Model](https://www.youtube.com/watch?v=gpsz8sc6mNU) - Valentin Ziegler at C++ Meeting 2014

## Learning Resources

#### Books

- [C++ Concurrency in Action: Practical Multithreading](https://www.manning.com/books/c-plus-plus-concurrency-in-action) - Anthony Williams 2012
- [The Art of Multiprocessor Programming](https://www.amazon.com/Art-Multiprocessor-Programming-Revised-Reprint/dp/0123973376/ref=sr_1_1?ie=UTF8&qid=1438003865&sr=8-1&keywords=maurice+herlihy) - Maurice Herlihy 2012
- [Parallel Computing: Theory and Practice](http://www.cs.cmu.edu/afs/cs/academic/class/15210-f15/www/tapp.html#ch:work-stealing) - Umut A. Acar 2016

#### Tutorials
- [MpiTutorial](mpitutorial.com) - A fantastic mpi tutorial
- [Beginners Guide to HPC](http://www.shodor.org/petascale/materials/UPModules/beginnersGuideHPC/)
- [Parallel Computing Training Tutorials](https://hpc.llnl.gov/training/tutorials) - Lawrence Livermore National Laboratory

#### Position Papers

- [The Landscape of Parallel Computing Research: A View from Berkeley](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf)
- [Extreme Heterogeneity 2018: Productive Computational Science in the Era of Extreme Heterogeneity](references/2018-Extreme-Heterogeneity-DoE.pdf)

#### Courses

- [Berkeley: Applications of Parallel Computers](https://sites.google.com/lbl.gov/cs267-spr2019/) - Detailed course on HPC
- [CS6290 High-performance Computer Architecture](https://www.udacity.com/course/high-performance-computer-architecture--ud007) - Milos Prvulovic and Catherine Gamboa at George Tech
- [Udacity High Performance Computing](https://www.youtube.com/playlist?list=PLAwxTw4SYaPk8NaXIiFQXWK6VPnrtMRXC)
- [Parallel Numerical Algorithms](https://solomonik.cs.illinois.edu/teaching/cs554/index.html)
- [Vanderbilt - Intro to HPC](https://github.com/vanderbiltscl/SC3260_HPC)
- [Illinois - Intro to HPC](https://andreask.cs.illinois.edu/Teaching/HPCFall2012/) - Creator of PyCuda

## Links

#### News
- [InsideHPC](https://insidehpc.com/)
- [HPCWire](https://www.hpcwire.com/)
- [NextPlatform](https://www.nextplatform.com)

#### Podcasts
- [This week in HPC](https://soundcloud.com/this-week-in-hpc)

#### Wikis
- [Supercomputer](https://en.wikipedia.org/wiki/Supercomputer)
- [Supercomputer architecture](https://en.wikipedia.org/wiki/Supercomputer_architecture)
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

#### Other
  - [Nice notes](https://caiorss.github.io/C-Cpp-Notes/Libraries.html)
  - [Pocket HPC Survival Guide](https://tin6150.github.io/psg/lsf.html)

## Acknowledgements

This repo started from the great curated list https://github.com/taskflow/awesome-parallel-computing


