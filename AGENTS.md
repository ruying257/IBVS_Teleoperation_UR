# AGENTS.md - Coding Guidelines for IBVS_Teleoperation_UR

## Project Overview

C++ robotics project for visual servoing and teleoperation with Universal Robots. Uses ViSP, OpenCV, TensorRT, and RealSense SDK.

## Build Commands

```bash
# Build the main project
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Build tests
cd test
cmake ../../test
make -j$(nproc)

# Run main executable
./IBVS_Teleoperation

# Run tests
cd test
./run_test.sh camera <model_path>          # Test with camera
./run_test.sh image <model_path> <image>   # Test with image
./run_test.sh video <model_path> <video>   # Test with video
```

## Code Style Guidelines

### Language Standard
- C++17 required
- Use standard library features where possible
- Prefer smart pointers over raw pointers (when C++11+ available)

### Naming Conventions
- **Classes**: PascalCase (e.g., `SystemController`, `TensorRTDetection`)
- **Structs**: PascalCase (e.g., `AppConfig`)
- **Methods/Functions**: camelCase (e.g., `initialize()`, `processIbvs()`)
- **Member Variables**: snake_case with trailing underscore or prefix
  - Private members: snake_case (e.g., `config`, `current_state`)
  - Pointer members: trailing underscore discouraged, prefer smart pointers
- **Local Variables**: snake_case (e.g., `velocity_send`, `force_z`)
- **Constants/Enums**: UPPER_CASE or PascalCase for enum values
  - Enum values: `STATE_IBVS`, `STATE_TELEOP`
- **Preprocessor Macros**: UPPER_CASE with H suffix for headers (e.g., `SYSTEMCONTROLLER_H`)
- **Template Parameters**: PascalCase

### File Organization
- Headers: `include/*.h`
- Sources: `src/*.cpp`
- Tests: `test/*.cpp`
- Models: `models/`
- Test data: `test_data/`

### Include Order
1. Corresponding header file (for .cpp files)
2. C/C++ standard library headers
3. Third-party library headers (ViSP, OpenCV, Eigen)
4. Project internal headers

Use angle brackets `<>` for external libraries, quotes `""` for project headers.

### Header Guards
Use include guards with H suffix:
```cpp
#ifndef FILENAME_H
#define FILENAME_H
// ... content ...
#endif // FILENAME_H
```

### Code Formatting
- Indent: 4 spaces (no tabs)
- Brace style: K&R (opening brace on same line)
- Maximum line length: ~100-120 characters
- Spaces around operators
- Comments: Use `//` for single line, `/* */` for file headers

### Error Handling
- Use try-catch blocks for exception-prone operations
- Check return values of critical functions
- Log errors to `std::cerr` with descriptive messages
- Use `EXIT_SUCCESS`/`EXIT_FAILURE` for return codes

### Comments
- File headers: Brief description in Chinese
- Method comments: Explain purpose and parameters
- Inline comments: Explain complex logic
- Avoid obvious comments

### Class Design
- Use RAII pattern for resource management
- Initialize members in constructor initialization lists
- Prefer composition over inheritance
- Mark single-argument constructors as explicit
- Use `const` correctness

### Memory Management
- Prefer `std::unique_ptr` and `std::shared_ptr`
- Use `std::make_unique`/`std::make_shared` for creation
- For raw pointers (legacy code): check for null before use
- Always release resources in destructors

### Dependencies
- ViSP: Visual servoing library (hard dependency)
- OpenCV: Computer vision (hard dependency)
- Eigen: Linear algebra (hard dependency)
- CUDA: Optional (for YOLO acceleration)
- TensorRT: Optional (for YOLO inference)
- RealSense SDK: Camera interface (hard dependency)

### Build System (CMake)
- Minimum version: 3.10
- Always check for dependencies with `find_package()`
- Use `target_link_libraries()` for linking
- Set `CMAKE_CXX_STANDARD 17` and `CMAKE_CXX_STANDARD_REQUIRED ON`

### Testing
- Tests are standalone executables, not using a test framework
- Run tests via `run_test.sh` script
- Tests support camera, image, and video inputs
- Test executable: `build/test/test_yolo_detector`

## Common Patterns

### State Machine
```cpp
enum State {
    STATE_IBVS,
    STATE_WAIT_SELECT,
    STATE_APPROACH,
    STATE_TELEOP
};
```

### Configuration Structure
```cpp
struct AppConfig {
    std::string robot_ip = "192.168.31.100";
    double tag_size = 0.03;
    // ... with default values
};
```

### Conditional Compilation
```cpp
#ifdef HAVE_TENSORRT
// TensorRT-specific code
#endif
```
