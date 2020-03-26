# CLIENT

## Build
1. Buy NVIDIA videocard
2. Install CUDA 10.2
3. Install OpenCV 4.2.0
4.  `git clone https://github.com/cpllbstr/Hist-Kalman-tracker`
5.  `cd Hist-Kalman-tracker`
6.  `cmake -S . -B ./build` - it will take some time to download and install dependencies
7.  `cd ./build`
8.  `make`

Binary file would be produced in the project's root dir.

After installation you need to set path to your darknet yolo cfg and weights in `config.toml`. 

## Launch 

To run correctly you need to set enviroment variable `DETECTOR_ADDR_PORT`.
It could be made by running this command: 

    export DETECTOR_ADDR_PORT="127.0.0.1:50051"

Then simply run `./client`. 
* WARNING!  Setting enviroment variable works only for current terminal. If you want to run it in another terminal, you need to set  `DETECTOR_ADDR_PORT` again.