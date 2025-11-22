.PHONY: all clean stems test

BUILD_DIR := build

all: $(BUILD_DIR)
	@cmake --build $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..

stems: all

clean:
	@rm -rf $(BUILD_DIR)

test: all
	@cd $(BUILD_DIR) && ctest --output-on-failure

.DEFAULT_GOAL := all
