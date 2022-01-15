#ifndef IPULIB_H
#define IPULIB_H

#include <stdexcept>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>

// This is a singleton based on: https://stackoverflow.com/a/1008289
class IPUContext {
private:
    poplar::Device device;
    poplar::Target target;

public:
    IPUContext() {
        auto manager = poplar::DeviceManager::createDeviceManager();
        // Attempt to attach to a single IPU:
        auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
        std::cout << "Trying to attach to IPU\n";
        auto it = std::find_if(devices.begin(), devices.end(), [](poplar::Device &device) {
           return device.attach();
        });

        if (it == devices.end()) {
          std::cerr << "Error attaching to device\n";
        }

        device = std::move(*it);
        target = device.getTarget();
    }


    poplar::Target& getTarget() {
        return target;
    }

    poplar::Device& getDevice() {
        return device;
    }

    poplar::Graph getGraph() {
        return std::move(poplar::Graph(target));
    }
};

#endif // IPULIB_H