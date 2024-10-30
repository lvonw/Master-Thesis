sudo apt-get upgrade

hostname -I for local IP

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

### Installing Conda 
nano ~/.bashrc
add EXPORT PATH="/home/paperspace/miniconda3/bin:$PATH"
source ~/.bashrc

conda init
source ~/.bashrc
conda create --name mtenv python=3.11 or lower

depending on nvcc --version
11.7
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

11.8 or higher 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install conda-forge::gdal
conda install matplotlib
conda install tqdm
conda install pyyaml (might not be necessary)

### Git
git clone https://github.com/lvonw/Master-Thesis.git
user 
access token

if running doesnt work then do whereis torchrun and copy the absolute path of 
the conda env

### Mounting
sudo chown paperspace:paperspace /home/paperspace/Master-Thesis/data
sudo nano /etc/fstab
//your-shared-drive-ip-address/your-shared-drive-name /home/paperspace/Master-Thesis/data   cifs  username=your-username,password=your-password,uid=1000,gid=1000,rw,user  0  0

verify with 
df -h