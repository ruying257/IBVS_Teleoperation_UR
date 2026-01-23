#include <iostream>

#include "AppConfig.h"
#include "SystemController.h"

int main(int argc, char **argv) {
    try {
        // 初始化配置
        AppConfig config;
        
        // 解析命令行参数（如果需要）
        // 这里可以添加命令行参数解析逻辑
        
        // 初始化系统控制器
        SystemController controller(config);
        
        // 初始化系统
        if (!controller.initialize()) {
            std::cerr << "系统初始化失败" << std::endl;
            return EXIT_FAILURE;
        }
        
        // 运行系统
        controller.run();
        
    } catch (const std::exception &e) {
        std::cout << "std::exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
