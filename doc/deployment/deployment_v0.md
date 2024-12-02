# Deployment v0

This document details how to manually configure a GPU worker node for training
with multiple GPUs. Currently this remains a manual process, however it might 
be updated at some point in the future to a more automatic apprach.

### Basics

Firstly connect to your machine using `ssh paperspace@[HOST-IP]`

The following command is rather slow and might not be necessary.

```
sudo apt-get update
```

If you are looking to work with multiple nodes you will need the IP address 
within the local network. You can find it using 

```
hostname -I for local IP
```

### Installing Conda 

The GDAL library is not available for download with PIP therefore it must be 
downloaded with conda. Conda is not preinstalled in the paperspace ML-in-a-box 
template and must therefore be downloaded manually. \
To do this run the following 4 commands sequentially

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

You will then have to configure the PATH environment variable to ensure the 
conda command is found in the terminal. To do this run the following command
in the Terminal which will open the NANO editor in which you can configure the
environment.

```
nano ~/.bashrc
```

Once opened add the following to the very end of the file, then press `ctrl+x` 
to close NANO, follow with `y` to save the changes and `enter` to apply the
changes to the current file.

```
add EXPORT PATH="/home/paperspace/miniconda3/bin:$PATH"
```
 
For a sanity check you can print the PATH variable using `echo $PATH`. After 
you have done this you will have to reload the path with

```
source ~/.bashrc
```

Now in order to be able to use conda effectively you will have to run the 
following 2 commands after which you should see `(base)` at the start of your
terminal prompt.

```
conda init
source ~/.bashrc
```

### Installing all dependancies with conda

Depending on the required CUDA version you will have to install either Python10
or Python11. You can check the CUDA version using `nvcc --version`. If the 
version is `11.7` or older you will have to go with Python10 otherwise 11 is 
fine. 

Create the conda environment using the following command

```
conda create --name mtenv python=3.11 or lower
```

Then depending on your CUDA version install one of the two PyTorch packages.
If the version doesn't match one of the templates, you may have to find the 
correct download on the PyTorch website.

```
11.7 or older
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

11.8 or higher 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Afterwards you will have to install these 4 packages. It is recommended that 
you install them in the same order as they are listed. (__Note__: Pyyaml may
be preinstalled depending on your PyTorch version)

```
conda install conda-forge::gdal
conda install matplotlib
conda install tqdm
conda install pyyaml
```

### Git

To clone the git repository use this command and verify using your GitHub
username and access token for the project

```
git clone https://github.com/lvonw/Master-Thesis.git
```

If you dont want to enter the credentials everytime you do a git action you can
cache them for a certain amount of seconds. Simply run this command once before
a git action

```
git config --global credential.helper 'cache --timeout=3600'
```

### Mounting

To mount the network drive you may first need to create the data directory with
`mkdir ./data`

Afterwards you edit the permission for mounting outside of the root directory
using 

```
sudo chown paperspace:paperspace /home/paperspace/Master-Thesis/data
```

Then you mount the drive by editing the fstab NANO

```
sudo nano /etc/fstab
```

Simply add this command at the bottom

```
//your-shared-drive-ip-address/your-shared-drive-name /home/paperspace/Master-Thesis/data   cifs  username=your-username,password=your-password,uid=1000,gid=1000,rw,user  0  0
```

Lastly now you can mount the directory you want with the call. Simply add the
same file you wrote into /etc/fstab 

```
mount /home/paperspace/Master-Thesis/data
```

You can verify whether this worked with  `df -h`

### Access Data

To download data from your machine run 

```
scp paperspace@host-ip:complete-host-path path-to-copy-to
```

Conversely to upload Data

```
scp path-to-upload paperspace@host-ip:complete-host-path 
```

To see the full path of a folder on your machine you can get the full path via 
`pwd`

If you uploaded an archive you extract the data via

```
tar -xzvf filename.tar.gz
```

### Running the Program

If running doesnt work then do `whereis torchrun` and copy the absolute path of 
the conda env instance of torchrun

This might require a higher file access limit which you can set with 
`ulimit -n 100000` 

You might even have to adjust the limit in `nano /etc/sysctl.conf`

Add this line to set the maximum number of open file descriptors:

```
fs.file-max = 12000
```
Save the file, then apply the changes with: `sudo sysctl -p`


### Monitoring

You can do `ls -l` to check when the files in the current directory where last 
changed 

You can see a file updating real time while looking at the bottom lines using
`tail -f -n 20 ./`

Check what processes are running with `htop`
