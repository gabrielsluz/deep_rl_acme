# Running Singularity on Mac
We need Vagrant!

The VM has access to the folder shared_dir.

After installing vagrant:
```
vagrant init sylabs/singularity-ce-3.9-ubuntu-bionic64
vagrant up
vagrant ssh
```

If already has a Vagrant VM:
```
vagrant destroy && \
    rm Vagrantfile
```