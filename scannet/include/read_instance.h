#pragma once

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <pcl/common/common.h>
#include <cmath>
#include <cstdint>
#include <utility>

struct CameraIntrinsics {
    float fx, fy, cx, cy;
};

struct Metadata {
    CameraIntrinsics depthIntrinsics;
    CameraIntrinsics colorIntrinsics;
    int depthWidth = 0, depthHeight = 0;
    int colorWidth = 0, colorHeight = 0;
    float depthShift = 1000.0f;
};

class InstanceCloudGenerator {
public:
    /// @brief Constructor
    /// @param infoFile Path to the info file containing the camera intrinsics and other metadata
    explicit InstanceCloudGenerator(const std::string& infoFile) {
        metadata_ = readMetadata(infoFile);
        std::cout << "Metadata read successfully" << std::endl;
    }

    /// @brief Process a single frame
    /// @param depthPath Path to the depth image
    /// @param instancePath Path to the instance image
    /// @param posePath Path to the pose file
    /// @param apply_filter Whether to apply a filter to the instance clouds
    /// @param global_frame Whether to use the global frame or the camera frame
    /// @param add_background Whether to add background points
    /// @param max_depth Maximum depth threshold in meters (points beyond this will be filtered out, 0 means no filtering)
    /// @param subsample_factor Subsample factor for rows and cols (1 = read all, 2 = read every other row/col, etc.)
    void processFrame(const std::string& depthPath,
                      const std::string& instancePath,
                      const std::string& posePath,
                      pcl::PointCloud<pcl::PointXYZRGB>& cloud_instances,
                      bool global_frame = true,
                      bool apply_filter = true,
                      bool add_background = false,
                      float max_depth = 0.0f,
                      int subsample_factor = 1) {
        cv::Mat depth = cv::imread(depthPath, cv::IMREAD_UNCHANGED);
        if (depth.empty()) {
            throw std::runtime_error(
                "Cannot read depth image: " + depthPath);
        }
        if (depth.type() != CV_16UC1) {
            throw std::runtime_error(
                "Depth image must be single-channel uint16: " + depthPath);
        }

        cv::Mat instance = cv::imread(instancePath, cv::IMREAD_UNCHANGED);
        if (instance.empty()) {
            throw std::runtime_error(
                "Cannot read instance image: " + instancePath);
        }
        if (instance.type() != CV_8UC1) {
            throw std::runtime_error(
                "Instance image must be single-channel uint8: " + instancePath);
        }

        Eigen::Matrix4f pose = loadPose(posePath);
        extractInstances(depth, instance, pose, cloud_instances, global_frame, apply_filter, add_background, max_depth, subsample_factor);
    }



    /// @brief Parse the intrinsics from the line
    /// @param line The line to parse
    /// @param intrinsics The intrinsics to parse
    static void parseIntrinsics(
        const std::string& line,
        CameraIntrinsics& intrinsics) {

        const auto equal_pos = line.find('=');
        if (equal_pos == std::string::npos) {
            throw std::runtime_error(
                "Missing '=' in intrinsic line: " + line);
        }

        // Read only the matrix elements after '='.
        std::istringstream iss(line.substr(equal_pos + 1));
        std::vector<float> values;
        float value;

        while (iss >> value) {
            values.push_back(value);
        }

        // Detect parsing errors such as non-numeric tokens or out-of-range values.
        if (!iss.eof()) {
            throw std::runtime_error(
                "Invalid numeric value in intrinsic line: " + line);
        }

        int stride;
        if (values.size() == 9) {
            stride = 3;
        } else if (values.size() == 16) {
            stride = 4;
        } else {
            throw std::runtime_error(
                "Expected 9 or 16 intrinsic values, got " +
                std::to_string(values.size()));
        }

        for (float v : values) {
            if (!std::isfinite(v)) {
                throw std::runtime_error(
                    "Intrinsic matrix contains non-finite values");
            }
        }

        CameraIntrinsics parsed{};
        parsed.fx = values[0];
        parsed.fy = values[stride + 1];
        parsed.cx = values[2];
        parsed.cy = values[stride + 2];

        if (parsed.fx <= 0.0f || parsed.fy <= 0.0f) {
            throw std::runtime_error(
                "Intrinsic focal lengths fx and fy must be positive");
        }

        intrinsics = parsed;
    }

    /// @brief Read the metadata from the file
    /// @param filename The filename to read
    /// @return The metadata
    static Metadata readMetadata(const std::string& filename) {
        Metadata meta{};
        std::ifstream file(filename);

        if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open camera metadata file: " + filename);
        }

        std::string line;
        bool has_depth_intrinsics = false;
        bool has_color_intrinsics = false;
        while (std::getline(file, line)) {
            if (line.find("m_depthWidth") == 0) meta.depthWidth = std::stoi(line.substr(line.find('=') + 1));
            else if (line.find("m_depthHeight") == 0) meta.depthHeight = std::stoi(line.substr(line.find('=') + 1));
            else if (line.find("m_colorWidth") == 0) meta.colorWidth = std::stoi(line.substr(line.find('=') + 1));
            else if (line.find("m_colorHeight") == 0) meta.colorHeight = std::stoi(line.substr(line.find('=') + 1));
            else if (line.find("m_depthShift") == 0) meta.depthShift = std::stof(line.substr(line.find('=') + 1));
            else if (line.find("m_calibrationColorIntrinsic") == 0) {
                parseIntrinsics(line, meta.colorIntrinsics);
                has_color_intrinsics = true;
            }
            else if (line.find("m_calibrationDepthIntrinsic") == 0) {
                parseIntrinsics(line, meta.depthIntrinsics);
                has_depth_intrinsics = true;
            }
        }
        // Require both intrinsic matrices before generating point clouds.
        if (!has_depth_intrinsics || !has_color_intrinsics) {
            throw std::runtime_error(
                "Missing depth or color intrinsics in metadata file: " +
                filename);
        }

        // The depth scale must be finite and strictly positive.
        if (!std::isfinite(meta.depthShift) || meta.depthShift <= 0.0f) {
            throw std::runtime_error(
                "Invalid depth scale in metadata file: " + filename);
        }
        std::cout << "Depth width: " << meta.depthWidth << std::endl;
        std::cout << "Depth height: " << meta.depthHeight << std::endl;
        std::cout << "Color width: " << meta.colorWidth << std::endl;
        std::cout << "Color height: " << meta.colorHeight << std::endl;
        std::cout << "Depth shift: " << meta.depthShift << std::endl;
        std::cout << "Color intrinsics: " << meta.colorIntrinsics.fx << " " << meta.colorIntrinsics.fy << " " << meta.colorIntrinsics.cx << " " << meta.colorIntrinsics.cy << std::endl;
        std::cout << "Depth intrinsics: " << meta.depthIntrinsics.fx << " " << meta.depthIntrinsics.fy << " " << meta.depthIntrinsics.cx << " " << meta.depthIntrinsics.cy << std::endl;
        return meta;
    }

    /// @brief Load the camera pose from the file
    /// @param posePath The path to the pose file
    /// @return The pose
    static Eigen::Matrix4f loadPose(const std::string& posePath) {
        std::ifstream file(posePath);
        if (!file.is_open()) {
            throw std::runtime_error(
                "Cannot open pose file: " + posePath);
        }

        Eigen::Matrix4f pose = Eigen::Matrix4f::Zero();

        // Read exactly 16 finite matrix elements.
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (!(file >> pose(i, j))) {
                    throw std::runtime_error(
                        "Cannot parse pose element (" +
                        std::to_string(i) + ", " +
                        std::to_string(j) + ") in: " + posePath);
                }

                if (!std::isfinite(pose(i, j))) {
                    throw std::runtime_error(
                        "Pose contains non-finite values: " + posePath);
                }
            }
        }

        // Reject extra content after the matrix, except whitespace.
        std::string extra;
        if (file >> extra) {
            throw std::runtime_error(
                "Unexpected content after pose matrix: " + posePath);
        }

        // Validate the homogeneous bottom row.
        constexpr float tolerance = 1e-5f;
        if (std::abs(pose(3, 0)) > tolerance ||
            std::abs(pose(3, 1)) > tolerance ||
            std::abs(pose(3, 2)) > tolerance ||
            std::abs(pose(3, 3) - 1.0f) > tolerance) {
            throw std::runtime_error(
                "Pose bottom row must be [0, 0, 0, 1]: " + posePath);
        }

        return pose;
    }


    /// @brief Apply the SOR filter to the point cloud
    /// @param cloud_in The input point cloud
    /// @param cloud_out The output point cloud
    void sorFilter(pcl::PointCloud<pcl::PointXYZRGB>& cloud_in, pcl::PointCloud<pcl::PointXYZRGB>& cloud_out, float mean_k = 50, float stddev_mul_thresh = 1.0) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_ptr(new pcl::PointCloud<pcl::PointXYZRGB>(cloud_in));
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud_in_ptr);
        sor.setMeanK(mean_k);  
        sor.setStddevMulThresh(stddev_mul_thresh);
        sor.filter(cloud_out);
    }


    /// @brief Apply the voxel filter to the point cloud
    /// @param cloud_in The input point cloud
    /// @param cloud_out The output point cloud
    // Store only occupied voxels and keep different labels separate.
    template <typename PointT, typename LabelGetter>
    void sparseVoxelFilter(
        const pcl::PointCloud<PointT>& cloud_in,
        pcl::PointCloud<PointT>& cloud_out,
        float leaf_size,
        LabelGetter get_label) {

        if (!std::isfinite(leaf_size) || leaf_size <= 0.0f) {
            throw std::invalid_argument(
                "Voxel leaf size must be finite and positive");
        }

        using Key = std::tuple<
            std::int64_t,
            std::int64_t,
            std::int64_t,
            std::uint32_t>;

        struct Accumulator {
            double sum_x = 0.0;
            double sum_y = 0.0;
            double sum_z = 0.0;
            std::size_t count = 0;
            PointT representative{};
        };

        std::map<Key, Accumulator> voxels;

        const double inverse_leaf =
            1.0 / static_cast<double>(leaf_size);

        const auto voxel_index = [inverse_leaf](float coordinate) {
            const double value = std::floor(
                static_cast<double>(coordinate) * inverse_leaf);

            // The upper bound is exclusive for signed 64-bit indices.
            const double limit = std::ldexp(1.0, 63);
            if (!std::isfinite(value) ||
                value < -limit || value >= limit) {
                throw std::overflow_error(
                    "Voxel coordinate exceeds the int64 range");
            }

            return static_cast<std::int64_t>(value);
        };

        for (const auto& point : cloud_in.points) {
            if (!std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                !std::isfinite(point.z)) {
                continue;
            }

            const Key key{
                voxel_index(point.x),
                voxel_index(point.y),
                voxel_index(point.z),
                get_label(point)
            };

            auto& cell = voxels[key];

            if (cell.count == 0) {
                cell.representative = point;
            }

            cell.sum_x += static_cast<double>(point.x);
            cell.sum_y += static_cast<double>(point.y);
            cell.sum_z += static_cast<double>(point.z);
            ++cell.count;
        }

        // Build a separate output to support in-place filtering safely.
        pcl::PointCloud<PointT> filtered;
        filtered.header = cloud_in.header;
        filtered.sensor_origin_ = cloud_in.sensor_origin_;
        filtered.sensor_orientation_ = cloud_in.sensor_orientation_;
        filtered.reserve(voxels.size());

        for (const auto& entry : voxels) {
            const auto& cell = entry.second;
            const double count = static_cast<double>(cell.count);

            PointT point = cell.representative;
            point.x = static_cast<float>(cell.sum_x / count);
            point.y = static_cast<float>(cell.sum_y / count);
            point.z = static_cast<float>(cell.sum_z / count);

            filtered.push_back(point);
        }

        filtered.is_dense = true;
        cloud_out = std::move(filtered);
    }

    // RGB channels encode instance IDs here, not display colors.
    void voxelFilter(
        pcl::PointCloud<pcl::PointXYZRGB>& cloud_in,
        pcl::PointCloud<pcl::PointXYZRGB>& cloud_out,
        float leaf_size = 0.02f) {

        sparseVoxelFilter(
            cloud_in,
            cloud_out,
            leaf_size,
            [](const pcl::PointXYZRGB& point) -> std::uint32_t {
                return
                    (static_cast<std::uint32_t>(point.r) << 16) |
                    (static_cast<std::uint32_t>(point.g) << 8) |
                    static_cast<std::uint32_t>(point.b);
            });
    }

    // XYZ clouds contain no label channels.
    void voxelFilter(
        pcl::PointCloud<pcl::PointXYZ>& cloud_in,
        pcl::PointCloud<pcl::PointXYZ>& cloud_out,
        float leaf_size = 0.05f) {

        sparseVoxelFilter(
            cloud_in,
            cloud_out,
            leaf_size,
            [](const pcl::PointXYZ&) -> std::uint32_t {
                return 0;
            });
    }

    /// @brief Extract the instances from the depth and instance images
    /// @param depth The depth image
    /// @param instance The instance image
    /// @param T The pose
    /// @param cloud_instances The instance cloud
    /// @param global_frame Whether to use the global frame or the camera frame
    /// @param apply_filter Whether to apply a filter to the instance clouds
    /// @param add_background Whether to add background points
    /// @param max_depth Maximum depth threshold in meters (points beyond this will be filtered out, 0 means no filtering)
    /// @param subsample_factor Subsample factor for rows and cols (1 = read all, 2 = read every other row/col, etc.)
    void extractInstances(const cv::Mat& depth,
                          const cv::Mat& instance,
                          const Eigen::Matrix4f& T,
                          pcl::PointCloud<pcl::PointXYZRGB>& cloud_instances,
                          bool global_frame = true,
                          bool apply_filter = true,
                          bool add_background = false,
                          float max_depth = 0.0f,
                          int subsample_factor = 1) {
                      
        float fx = metadata_.depthIntrinsics.fx, fy = metadata_.depthIntrinsics.fy;
        float cx = metadata_.depthIntrinsics.cx, cy = metadata_.depthIntrinsics.cy;
        float rgb_fx = metadata_.colorIntrinsics.fx, rgb_fy = metadata_.colorIntrinsics.fy;
        float rgb_cx = metadata_.colorIntrinsics.cx, rgb_cy = metadata_.colorIntrinsics.cy;
        float shift = metadata_.depthShift;

        pcl::PointCloud<pcl::PointXYZRGB> rawInstances;

        // Subsample the depth image by reading every n rows and cols
        for (int v = 0; v < depth.rows; v += subsample_factor) {
            for (int u = 0; u < depth.cols; u += subsample_factor) {
                uint16_t d = depth.at<uint16_t>(v, u);
                if (d == 0) continue;

                float z = d / shift;
                
                // Filter out depth values beyond the threshold
                if (max_depth > 0.0f && z > max_depth) continue;

                float x = (u - cx) * z / fx;
                float y = (v - cy) * z / fy;

                Eigen::Vector4f pt_cam(x, y, z, 1.0f);
                Eigen::Vector4f pt_world = T * pt_cam;

                // project to color image
                float u_rgb = x * rgb_fx / z + rgb_cx;
                float v_rgb = y * rgb_fy / z + rgb_cy;

                if (u_rgb >= 0 && u_rgb < instance.cols && v_rgb >= 0 && v_rgb < instance.rows) {
                    uint8_t instance_id = instance.at<uint8_t>(v_rgb, u_rgb);
                    if (instance_id == 0 && !add_background) continue;
                    
                    if (global_frame) {
                        pcl::PointXYZRGB pt;
                        pt.x = pt_world.x();
                        pt.y = pt_world.y();
                        pt.z = pt_world.z();
                        pt.r = instance_id;
                        pt.g = instance_id;
                        pt.b = instance_id;
                        rawInstances.push_back(pt);
                    } else {
                        pcl::PointXYZRGB pt;
                        pt.x = x;
                        pt.y = y;
                        pt.z = z;
                        pt.r = instance_id;
                        pt.g = instance_id;
                        pt.b = instance_id;
                        rawInstances.push_back(pt);
                    }
                    
                    // std::cout << "Added point " << pt_world.x() << ", " << pt_world.y() << ", " << pt_world.z() << " to instance " << static_cast<int>(instance_id) << std::endl;
                }
            }
        }

        if (apply_filter) {
            sorFilter(rawInstances, rawInstances);
            voxelFilter(rawInstances, rawInstances);
            cloud_instances = rawInstances;
        } else {
            cloud_instances = rawInstances;
        }
    }

private:
    Metadata metadata_;
};