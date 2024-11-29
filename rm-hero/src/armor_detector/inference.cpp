#include "../../include/armor_detector/inference.h"

static constexpr int INPUT_W = 416;    // Width of input
static constexpr int INPUT_H = 416;    // Height of input
static constexpr int NUM_CLASSES = 8;  // Number of classes
static constexpr int NUM_COLORS = 8;   // Number of color
static constexpr int TOPK = 128;       // TopK
static constexpr float NMS_THRESH = 0.3;
static constexpr float BBOX_CONF_THRESH = 0.85;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;

static inline int argmax(const float *ptr, int len)
    {
        int max_arg = 0;
        for (int i = 1; i < len; i++) {
            if (ptr[i] > ptr[max_arg]) max_arg = i;
        }
        return max_arg;
    }

/**
     * @brief Resize the image using letterbox
     * @param img Image before resize
     * @param transform_matrix Transform Matrix of Resize
     * @return Image after resize
     */
    inline cv::Mat scaledResize(cv::Mat& img, Eigen::Matrix<float,3,3> &transform_matrix)
    {
        float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
        int unpad_w = r * img.cols;
        int unpad_h = r * img.rows;

        int dw = INPUT_W - unpad_w;
        int dh = INPUT_H - unpad_h;

        dw /= 2;
        dh /= 2;

        transform_matrix << 1.0 / r, 0, -dw / r,
                            0, 1.0 / r, -dh / r,
                            0, 0, 1;

        cv::Mat re;
        cv::resize(img, re, cv::Size(unpad_w,unpad_h));
        cv::Mat out;
        cv::copyMakeBorder(re, out, dh, dh, dw, dw, cv::BORDER_CONSTANT);

        return out;
    }

    /**
         * @brief Generate grids and stride.
         * @param target_w Width of input.
         * @param target_h Height of input.
         * @param strides A vector of stride.
         * @param grid_strides Grid stride generated in this function.
         */
        static void generate_grids_and_stride(const int target_w, const int target_h,
                                                std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
        {
            for (auto stride : strides)
            {
                int num_grid_w = target_w / stride;
                int num_grid_h = target_h / stride;

                for (int g1 = 0; g1 < num_grid_h; g1++)
                {
                    for (int g0 = 0; g0 < num_grid_w; g0++)
                    {
                        GridAndStride grid_stride = {g0, g1, stride};
                        grid_strides.emplace_back(grid_stride);
                    }
                }
            }
        }

        /**
             * @brief Generate Proposal
             * @param grid_strides Grid strides
             * @param feat_ptr Original predition result.
             * @param prob_threshold Confidence Threshold.
             * @param objects Objects proposed.
             */
            static void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr,
                                                Eigen::Matrix<float,3,3> &transform_matrix,float prob_threshold,
                                                std::vector<ArmorObject>& objects)
            {
                const int num_anchors = grid_strides.size();
                //Travel all the anchors
                for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
                {
                    const int grid0 = grid_strides[anchor_idx].grid0;
                    const int grid1 = grid_strides[anchor_idx].grid1;
                    const int stride = grid_strides[anchor_idx].stride;
                    const int basic_pos = anchor_idx * (9 + (NUM_COLORS) + NUM_CLASSES);    // 8 + 8

                    // yolox/models/yolo_head.py decode logic
                    //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
                    //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
                    float x_1 = (feat_ptr[basic_pos + 0] + grid0) * stride;
                    float y_1 = (feat_ptr[basic_pos + 1] + grid1) * stride;
                    float x_2 = (feat_ptr[basic_pos + 2] + grid0) * stride;
                    float y_2 = (feat_ptr[basic_pos + 3] + grid1) * stride;
                    float x_3 = (feat_ptr[basic_pos + 4] + grid0) * stride;
                    float y_3 = (feat_ptr[basic_pos + 5] + grid1) * stride;
                    float x_4 = (feat_ptr[basic_pos + 6] + grid0) * stride;
                    float y_4 = (feat_ptr[basic_pos + 7] + grid1) * stride;

                    float box_objectness = (feat_ptr[basic_pos + 8]);
                    int box_color = argmax(feat_ptr + basic_pos + 9, NUM_COLORS);
                    int box_class = argmax(feat_ptr + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);

                    // cout << "output:" << endl;
                    // for (int ii = 0; ii < 25; ii++)
                    // {
                    //     cout << feat_ptr[basic_pos + ii] << " ";
                    // }
                    // cout << endl;
                    // float color_conf = (feat_ptr[basic_pos + 9 + box_color]);
                    // float cls_conf = (feat_ptr[basic_pos + 9 + NUM_COLORS + box_class]);
                    // float box_prob = (box_objectness + cls_conf + color_conf) / 3.0;
                    float box_prob = box_objectness;
                    if (box_prob >= prob_threshold)
                    {
                        ArmorObject obj;

                        Eigen::Matrix<float,3,4> apex_norm;
                        Eigen::Matrix<float,3,4> apex_dst;

                        apex_norm << x_1, x_2, x_3, x_4,
                                     y_1, y_2, y_3, y_4,
                                     1,   1,   1,   1;

                        apex_dst = transform_matrix * apex_norm;

                        for (int i = 0; i < 4; i++)
                        {
                            obj.apex[i] = cv::Point2f(apex_dst(0,i), apex_dst(1,i));
                            obj.pts.push_back(obj.apex[i]);
                        }

                        std::vector<cv::Point2f> tmp(obj.apex, obj.apex + 4);
                        // obj.rect = cv::boundingRect(tmp);
                        obj.cls = box_class;                                // 得分最大的那一个
                        obj.color = box_color;
                        obj.prob = box_prob;

                        // cout << "output:";
                        // for (int i = 0; i < 4; i++)
                        // {
                        //     cout << " " << "(" << obj.pts[i].x << "," << obj.pts[i].y << ") ";
                        // }
                        // cout << "obj_prob:" << obj.prob << " obj_color:" << obj.color << " obj_cls:" << obj.cls << endl;

                        objects.push_back(obj);
                    }
                } // point anchor loop
            }

            /**
                 * @brief Calculate intersection area between two objects.
                 * @param a Object a.
                 * @param b Object b.
                 * @return Area of intersection.
                 */
                static inline float intersection_area(const ArmorObject& a, const ArmorObject& b)
                {
                    cv::Rect_<float> inter = a.rect & b.rect;
                    return inter.area();
                }

                static void qsort_descent_inplace(std::vector<ArmorObject>& faceobjects, int left, int right)
                {
                    int i = left;
                    int j = right;
                    float p = faceobjects[(left + right) / 2].prob;

                    while (i <= j)
                    {
                        while (faceobjects[i].prob > p)
                            i++;

                        while (faceobjects[j].prob < p)
                            j--;

                        if (i <= j)
                        {
                            // swap
                            std::swap(faceobjects[i], faceobjects[j]);
                            i++;
                            j--;
                        }
                    }
                    // #pragma omp parallel sections
                    // {
                    //     #pragma omp section
                    //     {
                    //         if (left < j) qsort_descent_inplace(faceobjects, left, j);
                    //     }
                    //     #pragma omp section
                    //     {
                    //         if (i < right) qsort_descent_inplace(faceobjects, i, right);
                    //     }
                    // }
                    if (left < j) qsort_descent_inplace(faceobjects, left, j);
                    if (i < right) qsort_descent_inplace(faceobjects, i, right);
                }

                static void qsort_descent_inplace(std::vector<ArmorObject>& objects)
                    {
                        if (objects.empty())
                            return;

                        qsort_descent_inplace(objects, 0, objects.size() - 1);
                    }

                    static void nms_sorted_bboxes(std::vector<ArmorObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
                    {
                        picked.clear();
                        const int n = faceobjects.size();

                        std::vector<float> areas(n);
                        for (int i = 0; i < n; i++)
                        {
                            areas[i] = faceobjects[i].rect.area();
                        }

                        for (int i = 0; i < n; i++)
                        {
                            ArmorObject& a = faceobjects[i];
                            int keep = 1;
                            for (int j = 0; j < (int)picked.size(); j++)
                            {
                                ArmorObject& b = faceobjects[picked[j]];
                                // intersection over union
                                float inter_area = intersection_area(a, b);
                                float union_area = areas[i] + areas[picked[j]] - inter_area;
                                float iou = inter_area / union_area;
                                if (iou > nms_threshold || isnan(iou))
                                {
                                    keep = 0;
                                    //Stored for Merge
                                    if (iou > MERGE_MIN_IOU && abs(a.prob - b.prob) < MERGE_CONF_ERROR
                                                            && a.cls == b.cls && a.color == b.color)
                                    {
                                        for (int i = 0; i < 4; i++)
                                        {
                                            b.pts.push_back(a.apex[i]);
                                        }
                                    }
                                    // cout<<b.pts_x.size()<<endl;
                                }
                            }
                            if (keep)
                                picked.push_back(i);
                        }
                    }


                    /**
                         * @brief Decode outputs.
                         * @param prob Original predition output.
                         * @param objects Vector of objects predicted.
                         * @param img_w Width of Image.
                         * @param img_h Height of Image.
                         */
                        static void decodeOutputs(const float* prob, std::vector<ArmorObject>& objects, Eigen::Matrix<float, 3, 3> &transform_matrix)
                        {
                            std::vector<ArmorObject> proposals;
                            std::vector<int> strides = {8, 16, 32};
                            std::vector<GridAndStride> grid_strides;

                            generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides);
                            generateYoloxProposals(grid_strides, prob, transform_matrix, BBOX_CONF_THRESH, proposals);
                            qsort_descent_inplace(proposals);
                            if (proposals.size() >= TOPK)
                                proposals.resize(TOPK);
                            std::vector<int> picked;
                            nms_sorted_bboxes(proposals, picked, NMS_THRESH);
                            int count = picked.size();
                            objects.resize(count);
                            for (int i = 0; i < count; i++)
                            {
                                objects[i] = proposals[picked[i]];
                            }
                        }

                        float calcTriangleArea(cv::Point2f pts[3])
                        {
                            /**
                             * @brief 海伦公式计算三角形面积
                             *
                             * @param pts 三角形顶点
                             * @return float 面积
                             */
                            auto a = sqrt(pow((pts[0] - pts[1]).x, 2) + pow((pts[0] - pts[1]).y, 2));
                            auto b = sqrt(pow((pts[1] - pts[2]).x, 2) + pow((pts[1] - pts[2]).y, 2));
                            auto c = sqrt(pow((pts[2] - pts[0]).x, 2) + pow((pts[2] - pts[0]).y, 2));
                            auto p = (a + b + c) / 2.f;
                            return sqrt(p * (p - a) * (p - b) * (p - c));
                        }

                        float calcTetragonArea(cv::Point2f pts[4])
                        {
                            /**
                             * @brief 计算四边形面积
                             *
                             * @param pts 四边形顶点
                             * @return float 面积
                             */
                            return calcTriangleArea(&pts[0]) + calcTriangleArea(&pts[1]);
                        }

                        ArmorDetector::ArmorDetector()
                        {
                        }

                        ArmorDetector::~ArmorDetector()
                        {
                        }

                        bool ArmorDetector::initModel(std::string path)
                        {
                            std::cout<<"Start initialize model..."<<std::endl;
                            for(auto &device:core.get_available_devices())
                            {
                                std::cout<<"device:"<<device<<std::endl;
                            }

                            // Setting Configuration Values
                            core.set_property("CPU",ov::enable_profiling(true));

                            // Step 1.Create openvino runtime core
                            model=core.read_model(path);

                            // Preprocessing
                            ov::preprocess::PrePostProcessor ppp(model);
                            ppp.input().tensor().set_element_type(ov::element::f32);

                            // Set output precision
                            ppp.output().tensor().set_element_type(ov::element::f32);

                            // 将预处理融入原始模型
                            ppp.build();

                            // Step 2. Compile the model
                            compiled_model=core.compile_model(
                                model,
                                "CPU",
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
                             );

                        }
