#!/bin/bash

MUJOCO_DIR=~/.mujoco
mkdir -p $MUJOCO_DIR

# # 200
# wget https://roboti.us/download/mujoco200_linux.zip -O $MUJOCO_DIR/mujoco200.zip
# unzip $MUJOCO_DIR/mujoco200.zip -d  $MUJOCO_DIR/
# mv $MUJOCO_DIR/mujoco200_linux $MUJOCO_DIR/mujoco200
# wget https://roboti.us/file/mjkey.txt -P $MUJOCO_DIR
# rm $MUJOCO_DIR/mujoco200.zip

# Export the install path to bashrc or zshrc and source it if zsh
if [[ "$SHELL" == */zsh ]]; then
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$MUJOCO_DIR/mujoco200/bin" >> ~/.zshrc
    echo "Please restart your terminal or run 'source ~/.zshrc' to apply the changes."
else
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$MUJOCO_DIR/mujoco200/bin" >> ~/.bashrc
    source ~/.bashrc
fi