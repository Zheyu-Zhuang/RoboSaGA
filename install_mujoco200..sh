MUJOCO_DIR=~/.mujoco
mkdir -p $MUJOCO_DIR

# 200
wget https://roboti.us/download/mujoco200_linux.zip -O $MUJOCO_DIR/mujoco200.zip
unzip $MUJOCO_DIR/mujoco200.zip
wget https://roboti.us/file/mjkey.txt -P $MUJOCO_DIR
rm $MUJOCO_DIR/mujoco200.zip

# Export the install path to bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$MUJOCO_DIR/mujoco200_linux/bin" >> ~/.bashrc
