rmdir /Q /S build
mkdir build

cd ./build

cmake -G "Visual Studio 16 2019" -DCMAKE_CONFIGURATION_TYPES="Debug;Release" ..