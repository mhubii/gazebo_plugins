# Gazebo Plugins

The provided gazebo plugins work complementatory to the [gazebo_models](https://github.com/mhubii/gazebo_models). They add functions like keyboard navigation and autonomous control to the models.

## Build

You can decide which plugins to build and hence only require dependencies for specific plugins. To build use

```
mkdir build
cd build
```

The best way to preceede, is to use ccmake and edit the options on which plugins to build

```
ccmake ..
make
```

Alternatively, you can also use cmake and define flags. For example

```
cmake -DBUILD_VEHICLE_PLUGINS=ON ..
make
```

Once you have built the project, in order for Gazebo to find the generated libraries, you need edit the plugin path environment variable in your `~/.bashrc` as follows

```
export GAZEBO_PLUGIN_PATH=<folder to which you cloned this repository>/gazebo_plugins/build/lib:$GAZEBO_PLUGIN_PATH
```

## Vehicle Plugins

To build the vehicle plugins, add the flag `-DBUILD_VEHICLE_PLUGINS=ON` during the cmake step. This plugin depends on the [navigation](https://github.com/mhubii/navigation) library which enables keyboard control and autonomous control. Please follow the build steps there. If you installed the navigation library to a non default location, you need to adjust the `-DNAVIGATION_DIR=<location to which you installed it>`. Autonomous navigation is only possible with the C++ interface of [PyTorch](https://github.com/pytorch/pytorch). Simply define the option `-DBUILD_WITH_TORCH=OFF` to disable it.
