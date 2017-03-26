# pyseeta
python api for SeetaFaceEngine(https://github.com/seetaface/SeetaFaceEngine.git)
# installation
1. Download pyseeta(https://github.com/TuXiaokang/pyseeta.git)
2. `git submodule update --init --recursive`
3. Build the dynamic library.
>  ```bash
> cd SeetaFaceEngine/
> mkdir build; cd build
> cmake ..
> make  
> ```
4  Add the dynamic path to system environment variables.
> + on linux & macOS, the default is `SeetaFaceEngine/library`
> + on windows, the default is  `SeetaFaceEngine/library/release/`
5. install pyseeta
> ```bash
> python setup.py build
> python setup.py install
> ```
