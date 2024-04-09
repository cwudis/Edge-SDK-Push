
#ifndef __IMAGE_PROCESSOR_CAR_H__
#define __IMAGE_PROCESSOR_CAR_H__

#include <memory>

#include "image_processor.h"
#include "fastdeploy/vision.h"

namespace edge_app {

    class ImageProcessorCar : public ImageProcessor {
    public:
	std::string model_file = "/home/flision/cwudis/Edge-SDK-master/examples/common/data/car/inference.pdmodel";
        std::string params_file = "/home/flision/cwudis/Edge-SDK-master/examples/common/data/car/inference.pdiparams";
        std::string config_file = "/home/flision/cwudis/Edge-SDK-master/examples/common/data/car/inference.yml";
        ImageProcessorCar(const std::string& name) : show_name_(name),   model(model_file, params_file, config_file){}

        ~ImageProcessorCar() override {}

        int32_t Init() override;

        void Process(const std::shared_ptr<Image> image) override;
        
    private:
        std::string show_name_;
        
        enum {
            kFilePathSizeMax = 256,
            kCurrentFilePathSizeMax = 128,
        };
        char cur_file_dir_path_[kCurrentFilePathSizeMax];
        fastdeploy::vision::detection::PaddleDetectionModel model;
       // char model_file[kFilePathSizeMax];
       // char params_file[kFilePathSizeMax];
       // char config_file[kFilePathSizeMax];
    };

}  // namespace edge_app

#endif
