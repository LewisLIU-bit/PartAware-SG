/*
 * Author: Clarence Chen
 * This script is used to generate the ply map for a scene from the openset annotation.
 */

#include <string>
#include "read_instance.h"
#include <boost/filesystem.hpp>
#include <iomanip>
#include <json/single_include/nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <random>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <utility>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"
#include <pcl/kdtree/kdtree_flann.h>
namespace backward{
    backward::SignalHandling sh;
}

using json = nlohmann::json;
// Experimental spatial tolerance for instance association, in meters.
constexpr float kInstanceMatchRadius = 0.10f;
// Optional OP3DSG-inspired association; the default remains the saved baseline.
enum class AssociationMode { Legacy, OP3DSG };
AssociationMode association_mode = AssociationMode::Legacy;

struct FrameInput {
    std::string frame_id;
    std::string depth_path;
    std::string pose_path;
    std::string instance_path;
    std::string instance_json_path;
};

struct FloorSample {
    float x, y, z;
    bool background;
    std::size_t frame;
};

// Estimate a horizontal floor from multiple frames in a Z-up world.
// Return false when geometric/background support is insufficient.
bool estimateHorizontalFloor(
    const std::vector<FrameInput>& inputs,
    InstanceCloudGenerator& generator,
    float max_depth,
    int subsample_factor,
    float& floor_z
) {
    constexpr float bin_size = 0.02f;
    constexpr float fit_band = 0.01f;
    constexpr float cell_size = 0.10f;

    std::vector<FloorSample> samples;
    std::vector<float> heights;
    std::map<int, std::size_t> histogram;

    for (std::size_t frame = 0; frame < inputs.size(); ++frame) {
        const auto& input = inputs[frame];

        if (!boost::filesystem::exists(input.depth_path) ||
            !boost::filesystem::exists(input.pose_path) ||
            !boost::filesystem::exists(input.instance_path)) {
            std::cout << "[FLOOR_SCAN] Missing input: "
                      << input.frame_id << std::endl;
            continue;
        }

        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        // Use sparse pixels, world coordinates, and all valid labels.
        // Avoid SOR/voxel filtering during plane estimation.
        generator.processFrame(
            input.depth_path,
            input.instance_path,
            input.pose_path,
            cloud,
            true, false, true,
            max_depth,
            std::max(16, subsample_factor)
        );

        for (const auto& p : cloud.points) {
            if (!std::isfinite(p.x) ||
                !std::isfinite(p.y) ||
                !std::isfinite(p.z)) {
                continue;
            }

            samples.push_back({
                p.x, p.y, p.z,
                p.r == 0 && p.g == 0 && p.b == 0,
                frame
            });
            heights.push_back(p.z);
            ++histogram[static_cast<int>(std::floor(p.z / bin_size))];
        }

        if ((frame + 1) % 10 == 0 || frame + 1 == inputs.size()) {
            std::cout << "[FLOOR_SCAN] "
                      << frame + 1 << "/" << inputs.size()
                      << std::endl;
        }
    }

    if (samples.size() < 1000) {
        std::cout << "[FLOOR_SKIP] Too few valid samples." << std::endl;
        return false;
    }

    std::sort(heights.begin(), heights.end());

    // Restrict candidates to the low end of the observed scene.
    // This is an observed-height constraint, not an assumption that Z=0.
    const float low_height = heights[heights.size() / 50];

    for (const auto& entry : histogram) {
        const float center = (entry.first + 0.5f) * bin_size;

        if (center > low_height + 0.03f) {
            break;
        }
        if (entry.second < 100) {
            continue;
        }

        std::vector<float> local_heights;
        for (const auto& p : samples) {
            if (std::abs(p.z - center) <= 0.02f) {
                local_heights.push_back(p.z);
            }
        }

        if (local_heights.size() < 300) {
            continue;
        }

        auto middle = local_heights.begin() + local_heights.size() / 2;
        std::nth_element(
            local_heights.begin(), middle, local_heights.end()
        );
        const float candidate_z = *middle;

        std::size_t below = 0;
        std::size_t inliers = 0;
        std::set<std::pair<int, int>> background_cells;
        std::map<std::size_t, std::size_t> background_frame_counts;

        for (const auto& p : samples) {
            if (p.z < candidate_z - 0.03f) {
                ++below;
            }

            if (std::abs(p.z - candidate_z) > fit_band) {
                continue;
            }

            ++inliers;
            if (p.background) {
                background_cells.emplace(
                    static_cast<int>(std::floor(p.x / cell_size)),
                    static_cast<int>(std::floor(p.y / cell_size))
                );
                ++background_frame_counts[p.frame];
            }
        }

        std::size_t supported_frames = 0;
        for (const auto& item : background_frame_counts) {
            if (item.second >= 30) {
                ++supported_frames;
            }
        }

        const double below_fraction =
            static_cast<double>(below) / samples.size();

        // Require broad background support and repeated observations.
        if (inliers < 300 ||
            background_cells.size() < 100 ||
            supported_frames < 3 ||
            below_fraction > 0.02) {
            continue;
        }

        floor_z = candidate_z;
        std::cout << "[FLOOR_ACCEPT] z=" << floor_z
                  << " inliers=" << inliers
                  << " background_cells=" << background_cells.size()
                  << " supported_frames=" << supported_frames
                  << " below_fraction=" << below_fraction
                  << std::endl;
        return true;
    }

    std::cout << "[FLOOR_SKIP] No sufficiently supported horizontal floor."
              << std::endl;
    return false;
}

// Remove instance labels near the estimated floor while retaining geometry.
void clearFloorLabels(
    pcl::PointCloud<pcl::PointXYZRGB>& cloud,
    float floor_z,
    const std::string& frame_id
) {
    constexpr float removal_band = 0.01f;
    std::map<int, std::size_t> removed;

    for (auto& p : cloud.points) {
        if (p.r == 0 && p.g == 0 && p.b == 0) {
            continue;
        }
        if (!std::isfinite(p.z)) {
            continue;
        }

        if (std::abs(p.z - floor_z) <= removal_band) {
            ++removed[static_cast<int>(p.r)];
            p.r = 0;
            p.g = 0;
            p.b = 0;
        }
    }

    for (const auto& item : removed) {
        std::cout << "[FLOOR_CLEAR] frame=" << frame_id
                  << " local_id=" << item.first
                  << " points=" << item.second
                  << std::endl;
    }
}

std::vector<FrameInput> loadManifestFrames(
    const std::string& manifest_path,
    const std::string& instance_dir
) {
    namespace fs = boost::filesystem;

    std::ifstream file(manifest_path);
    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open manifest: " + manifest_path
        );
    }

    json manifest;
    file >> manifest;

    if (manifest.value("format", "") != "scannet_sg_input") {
        throw std::runtime_error("Expected a scannet_sg_input manifest");
    }

    const auto& frames = manifest.at("frames");
    if (!frames.is_array() || frames.empty()) {
        throw std::runtime_error("Manifest must contain non-empty frames");
    }

    const fs::path base_dir = fs::absolute(
        fs::path(manifest_path)
    ).parent_path();
    const fs::path labels_dir = fs::absolute(fs::path(instance_dir));

    auto resolve_input = [&base_dir](const std::string& value) {
        fs::path path(value);
        return path.is_absolute() ? path : base_dir / path;
    };

    std::vector<FrameInput> inputs;
    std::unordered_set<std::string> seen_ids;

    for (const auto& frame : frames) {
        const std::string id = frame.at("frame_id").get<std::string>();

        if (
            id.empty() || id == "." || id == ".."
            || id.find_first_of("/\\<>:\"|?*") != std::string::npos
            || id.back() == ' ' || id.back() == '.'
            || std::any_of(id.begin(), id.end(), [](unsigned char ch) {
                return ch < 32;
            })
        ) {
            throw std::runtime_error("Invalid frame ID: " + id);
        }

        if (!seen_ids.insert(id).second) {
            throw std::runtime_error("Duplicate frame ID: " + id);
        }

        FrameInput input;
        input.frame_id = id;
        input.depth_path = resolve_input(
            frame.at("depth").get<std::string>()
        ).string();
        input.pose_path = resolve_input(
            frame.at("pose").get<std::string>()
        ).string();
        // Accept both canonical frame IDs and legacy numeric filenames.
        auto resolve_label_path = [&](const std::string& suffix) {
            fs::path canonical_path = labels_dir / (id + suffix);

            if (fs::is_regular_file(canonical_path)) {
                return canonical_path;
            }

            const bool numeric_id = !id.empty() &&
                std::all_of(
                    id.begin(),
                    id.end(),
                    [](unsigned char c) {
                        return c >= '0' && c <= '9';
                    }
                );

            if (numeric_id) {
                std::string legacy_id = id;
                const auto first_nonzero =
                    legacy_id.find_first_not_of('0');

                if (first_nonzero == std::string::npos) {
                    legacy_id = "0";
                } else {
                    legacy_id = legacy_id.substr(first_nonzero);
                }

                fs::path legacy_path =
                    labels_dir / (legacy_id + suffix);

                if (fs::is_regular_file(legacy_path)) {
                    return legacy_path;
                }
            }

            return canonical_path;
        };

        input.instance_path =
            resolve_label_path(".png").string();

        input.instance_json_path =
            resolve_label_path("_instance.json").string();

        for (const auto& path : {
            input.depth_path,
            input.pose_path,
            input.instance_path,
            input.instance_json_path
        }) {
            if (!fs::is_regular_file(fs::path(path))) {
                throw std::runtime_error("Missing frame input: " + path);
            }
        }

        inputs.push_back(input);
    }

    return inputs;
}

struct InstanceInfo {
    std::string frame_id;
    int instance_id;
    int frame_instance_id;
    std::string object_name;
    std::string object_description;
    float confidence;
    std::vector<float> feature;
    std::vector<float> bert_embedding;
};

// Global instance id and accumulated points
struct GlobalInstance {
    int global_id;
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_points;
    std::vector<float> bert_embedding;
    std::vector<float> avg_feature;
    std::string object_name;
    std::string object_description;
    float confidence;
    int count;
    int largest_point_count;

    // Track distinct supporting frames separately from detection count.
    std::unordered_set<std::string> observed_frames;

    GlobalInstance() : 
        global_id(-1),
        accumulated_points(new pcl::PointCloud<pcl::PointXYZ>()),
        bert_embedding(),
        avg_feature(),
        object_name(),
        object_description(),
        confidence(0.0f),
        count(1),
        largest_point_count(0) {
    }
};

int next_global_id = 1;
std::vector<GlobalInstance> global_instances;
std::vector<std::vector<InstanceInfo>> updated_instance_frames;

// Cosine similarity of BERT features
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    if (a.size() != b.size()) {
        throw std::invalid_argument("Feature dimensions disagree during association");
    }
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-6);
}


std::vector<float> compute3DMIoU(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud1,
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud2,
    float dist_threshold = kInstanceMatchRadius)
{
    if (cloud1->empty() || cloud2->empty()) {
        return std::vector<float>{0.0f, 0.0f, 0.0f};
    }

    // Build KD-trees for both clouds
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1, kdtree2;
    kdtree1.setInputCloud(cloud1);
    kdtree2.setInputCloud(cloud2);

    std::vector<int> indices;
    std::vector<float> dists;

    int matched1_count = 0, matched2_count = 0;

    // Check each point in cloud1 against cloud2
    for (size_t i = 0; i < cloud1->size(); ++i) {
        if (kdtree2.radiusSearch(cloud1->at(i), dist_threshold, indices, dists) > 0) {
            matched1_count++;
        }
    }

    // Check each point in cloud2 against cloud1
    for (size_t i = 0; i < cloud2->size(); ++i) {
        if (kdtree1.radiusSearch(cloud2->at(i), dist_threshold, indices, dists) > 0) {
            matched2_count++;
        }
    }

    const float coverage1 =
        static_cast<float>(matched1_count) / cloud1->size();
    const float coverage2 =
        static_cast<float>(matched2_count) / cloud2->size();

    // Symmetric neighborhood coverage, not volumetric IoU.
    const float coverage_sum = coverage1 + coverage2;
    const float symmetric_coverage =
        coverage_sum > 0.0f
        ? 2.0f * coverage1 * coverage2 / coverage_sum
        : 0.0f;

    return std::vector<float>{
        symmetric_coverage,
        coverage1,
        coverage2
    };
}



int matchInstanceToGlobal(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& instance_cloud,
                         const std::vector<float>& embedding,
                         const std::vector<GlobalInstance>& global_instances,
                         float sim_threshold = 0.8f,
                         float miou_threshold = 0.3f,
                         const std::string& frame_id = "") {
    if (!instance_cloud || instance_cloud->points.size() < 5 || embedding.empty() || global_instances.empty()) {
        return -1;
    }

    for (const auto& pt : *instance_cloud) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            std::cerr << "Invalid point coordinates!" << std::endl;
            break;
        }
    }

    float best_score = -1.0f;
    int best_id = -1;

    for (const auto& global_inst : global_instances) {
        if (!global_inst.accumulated_points || global_inst.accumulated_points->empty()) {
            continue;
        }

        float sim = cosineSimilarity(
            embedding, global_inst.bert_embedding
        );


        if (sim < sim_threshold) {
            continue;
        }

        std::vector<float> miou = compute3DMIoU(
            instance_cloud, global_inst.accumulated_points
        );


        float score;
        if (association_mode == AssociationMode::OP3DSG) {
            if (global_inst.observed_frames.count(frame_id) != 0) continue;
            const float directional_overlap = std::max(miou[1], miou[2]);
            if (directional_overlap < 0.2f) continue;
            score = directional_overlap + (sim + 1.0f) / 2.0f;
            if (score < 1.2f) continue;
        } else {
            if (miou[0] < miou_threshold) continue;
            score = 0.5f * sim + 0.5f * miou[0];
        }
        if (score > best_score) {
            best_score = score;
            best_id = global_inst.global_id;
        }
    }

    return best_id;
}


void saveInstanceJson(const std::vector<InstanceInfo>& data, const std::string& filename) {
    // Serialize an empty result as [] instead of null.
    nlohmann::json j = nlohmann::json::array();
    for (const auto& inst : data) {
        if (inst.instance_id == -1) continue;
        nlohmann::json item;
        item["instance_id"] = inst.instance_id;
        item["frame_instance_id"] = inst.frame_instance_id;
        item["object_name"] = inst.object_name;
        item["object_description"] = inst.object_description;
        item["confidence"] = inst.confidence;
        item["feature"] = inst.feature;
        item["bert_embedding"] = inst.bert_embedding;
        j.push_back(item);
    }
    std::ofstream(filename) << j.dump(2);
}

void saveFeatureJson(const std::vector<GlobalInstance>& global_instances, const std::string& filename) {
    nlohmann::json j;
    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) continue;
        nlohmann::json item;
        item["instance_id"] = instance.global_id;
        item["occurance"] = instance.count;
        item["feature"] = instance.avg_feature;
        j.push_back(item);
    }
    std::ofstream(filename) << j.dump(2);
}

void saveBertEmbeddingJson(const std::vector<GlobalInstance>& global_instances, const std::string& filename) {
    nlohmann::json j;
    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) continue;
        std::string instance_id = std::to_string(instance.global_id);
        j[instance_id] = instance.bert_embedding;
    }
    std::ofstream(filename) << j.dump(2);
}

int main(int argc, char** argv) {
    // Remove the additive option before the unchanged positional parser.
    // PARTAWARE_ASSOCIATION marks this optional extension in source audits.
    std::vector<char*> positional;
    positional.push_back(argv[0]);
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == "--association_mode") {
            if (++index >= argc) throw std::invalid_argument("Missing association mode");
            const std::string mode(argv[index]);
            if (mode == "op3dsg") association_mode = AssociationMode::OP3DSG;
            else if (mode == "legacy") association_mode = AssociationMode::Legacy;
            else throw std::invalid_argument("Association mode must be legacy or op3dsg");
        } else {
            positional.push_back(argv[index]);
        }
    }
    argc = static_cast<int>(positional.size());
    argv = positional.data();
     // Default parameters
    std::string scene_name = "scene0702_00";
    int if_visualize = 1;
    float max_depth = 0.0f;  // 0 means no depth filtering
    int subsample_factor = 1;  // 1 means read all rows/cols
    bool manifest_mode = false;
    bool filter_floor = false;
    std::string manifest_path;
    std::string instance_dir;
    std::string output_ply_dir;
    std::string meta_file;

    std::string raw_images_parent_dir = "/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/images/scans/";
    std::string refined_instance_parent_dir = "/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/processed/openset_scans/";
  
    if (argc > 1 && std::string(argv[1]) == "--manifest") {
        if (argc != 6&& argc !=7) {
            std::cerr
                << "Usage: " << argv[0]
                << " --manifest <manifest.json> <processed_scene_dir>"
                << " <max_depth> <subsample_factor>[--fliter_floor]"
                << std::endl;
            return 1;
        }

        namespace fs = boost::filesystem;

        manifest_mode = true;
        if (argc == 7) {
            if (std::string(argv[6]) != "--filter_floor") {
                throw std::runtime_error(
                    "Unknown optional argument: " + std::string(argv[6])
                );
            }
            filter_floor = true;
        }
        if_visualize = 0;
        manifest_path = fs::absolute(fs::path(argv[2])).string();

        const fs::path processed_scene_dir =
            fs::absolute(fs::path(argv[3]));

        instance_dir = (
            processed_scene_dir / "refined_instance"
        ).string();
        output_ply_dir = processed_scene_dir.string() + "/";

        max_depth = std::stof(argv[4]);
        subsample_factor = std::stoi(argv[5]);

        std::ifstream manifest_file(manifest_path);
        if (!manifest_file.is_open()) {
            throw std::runtime_error(
                "Cannot open manifest: " + manifest_path
            );
        }

        json manifest;
        manifest_file >> manifest;

        if (manifest.value("format", "") != "scannet_sg_input") {
            throw std::runtime_error(
                "Expected a scannet_sg_input manifest"
            );
        }

        scene_name = manifest.at("scene_id").get<std::string>();
        // Limit this experimental filter to the verified input convention.
        if (filter_floor &&
            (manifest.value("dataset", "") != "hypersim" ||
             manifest.value("world_frame", "") != "hypersim_world_z_up" ||
             manifest.value("length_unit", "") != "meter")) {
            throw std::runtime_error(
                "--filter_floor requires Hypersim, Z-up world coordinates, "
                "and meter units."
            );
        }

        const fs::path manifest_dir =
            fs::path(manifest_path).parent_path();
        fs::path camera_info =
            manifest.at("camera_info").get<std::string>();

        if (!camera_info.is_absolute()) {
            camera_info = manifest_dir / camera_info;
        }

        if (!fs::is_regular_file(camera_info)) {
            throw std::runtime_error(
                "Missing camera metadata: " + camera_info.string()
            );
        }

        if (!fs::is_directory(fs::path(instance_dir))) {
            throw std::runtime_error(
                "Missing instance directory: " + instance_dir
            );
        }

        meta_file = camera_info.string();

    }else if (argc == 1) {
        // Preserve the legacy default mode.
    }else if (argc == 2) {
        scene_name = argv[1];
    }else if (argc == 3) {
        scene_name = argv[1];
        if_visualize = std::stoi(argv[2]);
    }else if (argc == 5) {
        scene_name = argv[1];
        if_visualize = std::stoi(argv[2]);
        refined_instance_parent_dir = argv[3];
        raw_images_parent_dir = argv[4];
    }else if (argc == 7) {
        scene_name = argv[1];
        if_visualize = std::stoi(argv[2]);
        refined_instance_parent_dir = argv[3];
        raw_images_parent_dir = argv[4];
        max_depth = std::stof(argv[5]);
        subsample_factor = std::stoi(argv[6]);
    }else{
        std::cout << "Usage: " << argv[0] << " <scene_name> <if_visualize 0 or 1> <refined_instance_parent_dir> <raw_images_parent_dir> [max_depth] [subsample_factor]" << std::endl;
        std::cout << "  max_depth: Maximum depth threshold in meters (0 = no filtering)" << std::endl;
        std::cout << "  subsample_factor: Subsample factor for rows/cols (1 = read all, 2 = read every other row/col, etc.)" << std::endl;
        return 1;
    }

    // Add "/" to the end of the refined_instance_parent_dir if not exists
    if (refined_instance_parent_dir.back() != '/') {
        refined_instance_parent_dir += "/";
    }
    if (raw_images_parent_dir.back() != '/') {
        raw_images_parent_dir += "/";
    }

    std::cout << "refined_instance_parent_dir: " << refined_instance_parent_dir << std::endl;
    std::cout << "raw_images_parent_dir: " << raw_images_parent_dir << std::endl;
    std::cout << "scene_name: " << scene_name << std::endl;
    std::cout << "if_visualize: " << if_visualize << std::endl;
    std::cout << "max_depth: " << max_depth << " meters (0 = no filtering)" << std::endl;
    std::cout << "subsample_factor: " << subsample_factor << std::endl;

    if (!manifest_mode) {
        output_ply_dir = refined_instance_parent_dir + scene_name + "/";
        instance_dir = output_ply_dir + "refined_instance";
        meta_file = raw_images_parent_dir + scene_name + "/_info.txt";
    }

    if (!std::isfinite(max_depth) || max_depth < 0.0f) {
        throw std::runtime_error(
            "max_depth must be finite and non-negative"
        );
    }

    if (subsample_factor < 1) {
        throw std::runtime_error(
            "subsample_factor must be at least 1"
        );
    }

    std::cout << "Input mode: "
              << (manifest_mode ? "manifest" : "legacy")
              << std::endl;
    std::cout << "Camera metadata: " << meta_file << std::endl;
    std::cout << "Instance directory: " << instance_dir << std::endl;
    std::cout << "Output directory: " << output_ply_dir << std::endl;

    InstanceCloudGenerator cloud_generator(meta_file);

    std::vector<FrameInput> frame_inputs;

    if (manifest_mode) {
        frame_inputs = loadManifestFrames(manifest_path, instance_dir);
    } else {
        // Preserve numeric ordering and filenames for legacy ScanNet input.
        std::vector<int> frame_id_list;

        for (const auto& entry :
             boost::filesystem::directory_iterator(instance_dir)) {
            if (
                boost::filesystem::is_regular_file(entry.path())
                && entry.path().extension() == ".png"
            ) {
                frame_id_list.push_back(
                    std::stoi(entry.path().stem().string())
                );
            }
        }

        std::sort(frame_id_list.begin(), frame_id_list.end());

        for (int frame_id : frame_id_list) {
            std::ostringstream padded;
            padded << std::setw(6) << std::setfill('0') << frame_id;

            const std::string legacy_id = std::to_string(frame_id);
            const std::string raw_prefix =
                raw_images_parent_dir + scene_name
                + "/frame-" + padded.str();

            FrameInput input;
            input.frame_id = legacy_id;
            input.depth_path = raw_prefix + ".depth.pgm";
            input.pose_path = raw_prefix + ".pose.txt";
            input.instance_path = (
                boost::filesystem::path(instance_dir)
                / (legacy_id + ".png")
            ).string();
            input.instance_json_path = (
                boost::filesystem::path(instance_dir)
                / (legacy_id + "_instance.json")
            ).string();

            frame_inputs.push_back(input);
        }
    }

    if (frame_inputs.empty()) {
        throw std::runtime_error("No frames available for reconstruction");
    }

    std::cout << "Frames to process: " << frame_inputs.size()
              << std::endl;
    float floor_z = 0.0f;
    bool floor_ready = false;

    if (filter_floor) {
        floor_ready = estimateHorizontalFloor(
            frame_inputs,
            cloud_generator,
            max_depth,
            subsample_factor,
            floor_z
        );
    }
    pcl::PointCloud<pcl::PointXYZRGB> background_accumulated;
    int background_voxel_frame_counter = 0;

    // Refresh one progress line after each work item, including skipped frames.
    auto report_progress = [](const char* stage, std::size_t completed,
                              std::size_t total) {
        const double fraction = total == 0 ? 1.0
            : static_cast<double>(completed) / total;
        const int width = 30;
        const int filled = static_cast<int>(fraction * width);
        std::ostringstream line;
        line << '\r' << stage << " [" << std::string(filled, '=');
        if (filled < width) {
            line << '>' << std::string(width - filled - 1, ' ');
        }
        line << "] " << std::fixed << std::setprecision(1)
             << fraction * 100.0 << "% (" << completed << "/" << total << ")";
        std::cout << line.str() << std::flush;
        if (completed == total) std::cout << std::endl;
    };
    std::size_t visited_frames = 0;
    
    // Process both input modes through the same reconstruction pipeline.
    for (const auto& input : frame_inputs) {
        report_progress("Frame fusion", visited_frames, frame_inputs.size());
        ++visited_frames;

        const std::string& depth_path = input.depth_path;
        const std::string& instance_path = input.instance_path;
        const std::string& pose_path = input.pose_path;
        const std::string& json_path = input.instance_json_path;
        
        if (!boost::filesystem::exists(instance_path)) {
            std::cout << "*** Instance file does not exist: " << instance_path << "Will try the next frame" << std::endl;
            continue;
        }
        
        if (!boost::filesystem::exists(depth_path)) {
            std::cout << "*** Depth file does not exist: " << depth_path << std::endl;
            return 1;
        }
        
        if (!boost::filesystem::exists(pose_path)) {
            std::cout << "*** Pose file does not exist: " << pose_path << std::endl;
            return 1;
        }

        if (!boost::filesystem::exists(json_path)) {
            std::cout << "*** JSON file does not exist: " << json_path << std::endl;
            return 1;
        }

        pcl::PointCloud<pcl::PointXYZRGB> cloud_instances;
        cloud_generator.processFrame(depth_path, instance_path, pose_path, cloud_instances, true, true, true, max_depth, subsample_factor);
        // Clear floor contamination before instance extraction and association.
        if (floor_ready) {
            clearFloorLabels(cloud_instances, floor_z, input.frame_id);
        }
        std::size_t labeled_points = 0;
        for (const auto& point : cloud_instances.points) {
            if (point.r != 0) {
                ++labeled_points;
            }
        }

        for (const auto& pt : cloud_instances.points) {
            if (pt.r == 0 && pt.g == 0 && pt.b == 0) {
                background_accumulated.points.push_back(pt);
            }
        }
        background_voxel_frame_counter++;
        if (background_voxel_frame_counter % 10 == 0 && !background_accumulated.points.empty()) {
            background_accumulated.width = static_cast<uint32_t>(background_accumulated.points.size());
            background_accumulated.height = 1;
            background_accumulated.is_dense = true;
            pcl::PointCloud<pcl::PointXYZRGB> bg_filtered;
            cloud_generator.voxelFilter(background_accumulated, bg_filtered);
            background_accumulated = bg_filtered;
        }

        // Load the json file
        std::ifstream json_file(json_path);
        json json_data;
        json_file >> json_data;

        std::vector<InstanceInfo> instances;
        for (const auto& item : json_data) {
            InstanceInfo inst;
            inst.instance_id = item["instance_id"];
            inst.frame_instance_id = item["frame_instance_id"];
            inst.object_name = item["object_name"];
            inst.object_description = item["object_description"];
            inst.confidence = item.value("confidence", 0.0f);
            inst.feature = item.value("feature", std::vector<float>{});
            inst.bert_embedding = item.value("bert_embedding", std::vector<float>{});
            inst.frame_id = input.frame_id;
            instances.push_back(inst);
        }

        // Process the instances one by one
        for (auto& inst : instances) {
            // Extract point cloud segment for this instance
            pcl::PointCloud<pcl::PointXYZ>::Ptr instance_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& pt : cloud_instances) {
                if (pt.r == inst.frame_instance_id) { //In one frame, Max id is 255
                    instance_cloud->push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
                }
            }
            if (!instance_cloud) {
                std::cerr << "instance_cloud is null!" << std::endl;
                continue;
            }
            if (instance_cloud->empty()) continue;
            if (inst.bert_embedding.empty()) continue;

            // Try to match with global instances
            /// TODO: Tune the parameters for matching
                        // Probe spatial tolerance on the second manifest frame only.
            if (
                manifest_mode
                && frame_inputs.size() > 1
                && input.frame_id == frame_inputs[1].frame_id
            ) {
                for (const auto& candidate : global_instances) {
                    if (
                        candidate.global_id == -1
                        || candidate.object_name != inst.object_name
                        || !candidate.accumulated_points
                        || candidate.accumulated_points->empty()
                    ) {
                        continue;
                    }


                    for (float radius : {0.05f, 0.10f, 0.20f}) {
                        compute3DMIoU(
                            instance_cloud,
                            candidate.accumulated_points,
                            radius
                        );
                    }
                }
            }
            int matched_id = matchInstanceToGlobal(instance_cloud, inst.bert_embedding, global_instances,
                                                   0.8f, 0.3f, input.frame_id);

            if (matched_id == -1) {
                // No match → assign new global ID
                GlobalInstance new_global;
                new_global.global_id = next_global_id++;
                new_global.accumulated_points->points = instance_cloud->points;
                new_global.bert_embedding = inst.bert_embedding;
                new_global.avg_feature = inst.feature;
                new_global.object_name = inst.object_name;
                new_global.object_description = inst.object_description;
                new_global.confidence = inst.confidence;
                new_global.count = 1;
                new_global.observed_frames.insert(input.frame_id);
                new_global.largest_point_count = instance_cloud->points.size();
                global_instances.push_back(new_global);
                matched_id = new_global.global_id;

                if (if_visualize) std::cout << "*** No match → assign new global ID: " << new_global.global_id << std::endl;
            } else {
                // Update existing global instance by adding the instance points to the global instance
                for (auto& g : global_instances) {
                    if (g.global_id == matched_id) {
                        for (const auto& pt : instance_cloud->points) {
                            g.accumulated_points->points.push_back(pt);
                        }
                        // Update the avg visual language feature by averaging the feature of the instance
                        for (size_t i = 0; i < g.avg_feature.size(); ++i) {
                            g.avg_feature[i] = (g.avg_feature[i] * g.count + inst.feature[i]) / (g.count + 1);
                        }
                        // Update the bert embedding and object name, description, confidence if the instance has more points
                        if (g.largest_point_count < instance_cloud->points.size()) {
                            g.largest_point_count = instance_cloud->points.size();
                            g.bert_embedding = inst.bert_embedding;
                            g.object_name = inst.object_name;
                            g.object_description = inst.object_description;
                            g.confidence = inst.confidence;
                        }

                        g.count += 1;
                        g.observed_frames.insert(input.frame_id);

                        // Downsample only the instance updated by this observation.
                        if (g.accumulated_points &&
                            g.accumulated_points->points.size() > 10000) {
                            pcl::PointCloud<pcl::PointXYZ> downsampled_cloud;
                            downsampled_cloud.reserve(
                                (g.accumulated_points->points.size() + 1) / 2);

                            for (size_t i = 0;
                                 i < g.accumulated_points->points.size();
                                 i += 2) {
                                downsampled_cloud.push_back(
                                    g.accumulated_points->points[i]);
                            }

                            downsampled_cloud.is_dense = g.accumulated_points->is_dense;
                            *g.accumulated_points = downsampled_cloud;
                        }

                        break;
                    }


                }
                if (if_visualize) std::cout << "*** Update existing global instance: " << matched_id << std::endl;
            }

            // Update the instance id
            inst.instance_id = matched_id;
        }

        // Update the updated_instance_frames
        updated_instance_frames.push_back(instances);
    }
    report_progress("Frame fusion", visited_frames, frame_inputs.size());

    // Compute the center of the global instances for quick filtering
    std::vector<float> global_instances_center_x;
    std::vector<float> global_instances_center_y;
    std::vector<float> global_instances_center_z;
    for (auto& instance : global_instances) {
        float center_x = 0.0f;
        float center_y = 0.0f;
        float center_z = 0.0f;
        for (auto& pt : *instance.accumulated_points) {
            center_x += pt.x;
            center_y += pt.y;
            center_z += pt.z;   
        }
        center_x /= instance.accumulated_points->points.size();
        center_y /= instance.accumulated_points->points.size();
        center_z /= instance.accumulated_points->points.size();
        global_instances_center_x.push_back(center_x);
        global_instances_center_y.push_back(center_y);
        global_instances_center_z.push_back(center_z);
    }

    // Further merge the global instances by cosine similarity and 3D MIoU
    if (if_visualize) std::cout << "*** Further merge the global instances by cosine similarity and 3D MIoU" << std::endl;
    
    std::unordered_map<int, int> merge_map;
    for (size_t i = 0; i < global_instances.size(); ++i) {
        report_progress("Instance merge", i, global_instances.size());
        GlobalInstance& instance = global_instances[i];
        for (size_t j = 0; j < global_instances.size(); ++j) {
            if (i==j) continue;

            GlobalInstance& other_instance = global_instances[j];
            if (instance.global_id == other_instance.global_id || other_instance.global_id == -1 || instance.global_id == -1) continue;

            float distance = std::sqrt(
                (global_instances_center_x[i] - global_instances_center_x[j]) * (global_instances_center_x[i] - global_instances_center_x[j]) +
                (global_instances_center_y[i] - global_instances_center_y[j]) * (global_instances_center_y[i] - global_instances_center_y[j]) +
                (global_instances_center_z[i] - global_instances_center_z[j]) * (global_instances_center_z[i] - global_instances_center_z[j])
            );
            if (distance > 2.0f) continue; // Filter out the instances that are too far away to accelerate the merging

            float cosine_similarity = cosineSimilarity(instance.bert_embedding, other_instance.bert_embedding);
            std::vector<float> miou = compute3DMIoU(
                instance.accumulated_points,
                other_instance.accumulated_points,
                kInstanceMatchRadius
            );
            // Preserve the existing merge rule.
            const bool original_merge =
                miou[0] > 0.7f ||
                (cosine_similarity > 0.7f &&
                 (miou[0] > 0.3f || miou[1] > 0.8f || miou[2] > 0.8f));

            // Check whether both tracks contain observations from the same frame.
            bool shared_frame = false;
            for (const auto& frame_id : instance.observed_frames) {
                if (other_instance.observed_frames.count(frame_id) != 0) {
                    shared_frame = true;
                    break;
                }
            }

            const float lower_coverage = std::min(miou[1], miou[2]);
            const float higher_coverage = std::max(miou[1], miou[2]);

            // Experimental recovery for geometrically overlapping tracks
            // whose category embeddings are moderately similar.
            // Require support from multiple frames on both sides.
            const bool geometry_recovery =
                !shared_frame &&
                instance.observed_frames.size() >= 2 &&
                other_instance.observed_frames.size() >= 2 &&
                cosine_similarity >= 0.50f &&
                lower_coverage >= 0.50f &&
                higher_coverage >= 0.80f;

            if (!original_merge && geometry_recovery) {
                std::cout
                    << "[GEOMETRY_RECOVERY] "
                    << instance.global_id << ":" << instance.object_name
                    << " <- "
                    << other_instance.global_id << ":" << other_instance.object_name
                    << " text=" << cosine_similarity
                    << " coverage=" << miou[1] << "," << miou[2]
                    << std::endl;
            }

const bool should_merge = association_mode == AssociationMode::Legacy
                ? (original_merge || geometry_recovery)
                : (!shared_frame && cosine_similarity >= 0.8f &&
                   higher_coverage >= 0.7f && lower_coverage >= 0.1f);
            if (should_merge) {
                if (if_visualize) {
                    std::cout << "*** Merging the two instances: " << instance.global_id << " and " << other_instance.global_id << std::endl;
                    std::cout << "*** The merging instance name is " << instance.object_name << " with " << other_instance.object_name << std::endl;
                }
                
                // Merge the two instances
                for (auto& pt : other_instance.accumulated_points->points) {
                    instance.accumulated_points->points.push_back(pt);
                }   
                
                // Update the bert embedding and object name, description, confidence if the instance has more points
                if (instance.largest_point_count < other_instance.largest_point_count) {
                    instance.bert_embedding = other_instance.bert_embedding;
                    instance.object_name = other_instance.object_name;
                    instance.object_description = other_instance.object_description;
                    instance.confidence = other_instance.confidence;
                }
                if (if_visualize) std::cout << "merged instance name: " << instance.object_name << std::endl;

                // Update the avg visual language feature by averaging the feature of the instance
                for (size_t i = 0; i < instance.avg_feature.size(); ++i) {
                    instance.avg_feature[i] = (instance.avg_feature[i] * instance.count + other_instance.avg_feature[i]) / (instance.count + 1);
                }

                instance.count += other_instance.count;
                instance.observed_frames.insert(
                    other_instance.observed_frames.begin(),
                    other_instance.observed_frames.end()
                );
                instance.largest_point_count = std::max(instance.largest_point_count, other_instance.largest_point_count);
                merge_map[other_instance.global_id] = instance.global_id;
                other_instance.global_id = -1;
            }
        }
    }

    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) {
            continue;
        }

    }
    report_progress("Instance merge", global_instances.size(),
                    global_instances.size());
    // Remove the global instances that is observed less than 2 times
    if (if_visualize) std::cout << "*** Remove the global instances that is observed less than 2 times" << std::endl;
    int removed_instances = 0;
    for (auto& instance : global_instances) {
        if (instance.global_id == -1) {
            continue;
        }

        const std::size_t frame_count = instance.observed_frames.size();

        if (frame_count < 2) {
            std::cout
                << "[LOW_FRAME_SUPPORT] id=" << instance.global_id
                << " name=" << instance.object_name
                << " detections=" << instance.count
                << " distinct_frames=" << frame_count
                << std::endl;

            merge_map[instance.global_id] = -1;
            instance.global_id = -1;
            ++removed_instances;
        }
    }

    // std::cout << "*** Global instances size: " << global_instances.size() << std::endl;
    // std::cout << "*** Removed " << removed_instances << " instances" << std::endl;

    // Update the instance id in the updated_instance_frames
    if (if_visualize) std::cout << "*** Update the instance id in the updated_instance_frames" << std::endl;
    for (auto& frame : updated_instance_frames) {
        for (auto& inst : frame) {
            // If the instance id is in the merge_map, update it iteratively until it is not in the merge_map
            int id = inst.instance_id;
            while (merge_map.find(id) != merge_map.end()) {
                int new_id = merge_map[id];
                id = new_id;
                if (new_id == -1) break;
                // std::cout << "Update the instance id: " << id << " to " << new_id << std::endl;
            }
            inst.instance_id = id;
        }
    }

    // Save the instance name map to a csv
    std::ofstream instance_name_map_file(output_ply_dir + "instance_name_map.csv");
    instance_name_map_file << "instance_id,name" << std::endl;
    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) continue;
        instance_name_map_file << instance.global_id << "," << instance.object_name << std::endl;
    }
    instance_name_map_file.close();

    // Save the instance id, count, and visual language feature to a json file
    if (if_visualize) std::cout << "*** Save the instance id, count, and visual language feature to a json file" << std::endl;
    saveFeatureJson(global_instances, output_ply_dir + "averaged_instance_features.json");
    saveBertEmbeddingJson(global_instances, output_ply_dir + "instance_bert_embeddings.json");

    // Save the updated_instance_frames to a new json file by each frame
    if (if_visualize) std::cout << "*** Save the updated_instance_frames to a new json file by each frame" << std::endl;
    for (auto& frame : updated_instance_frames) {
        if (frame.empty()) continue;
        std::string save_path = (
            boost::filesystem::path(instance_dir)
            / (frame[0].frame_id + "_updated_instance.json")
        ).string();
        saveInstanceJson(frame, save_path);
    }
    
    // Make a ply file for the global instances
    pcl::PointCloud<pcl::PointXYZRGB> global_instances_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> global_instances_cloud_random_color;
    std::vector<int> global_instances_id_list;
    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) continue;
        int id = instance.global_id;
        // rgb color is id % 255, id / 255 % 255, id / 255 / 255 % 255
        int r = id % 255;
        int g = (id / 255) % 255;
        int b = (id / 255 / 255) % 255;
        // Random color
        int r_random = rand() % 255;
        int g_random = rand() % 255;
        int b_random = rand() % 255;
        for (const auto& pt : *instance.accumulated_points) {
            pcl::PointXYZRGB p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            p.r = r;
            p.g = g;
            p.b = b;
            global_instances_cloud.points.push_back(p);

            // Random color
            p.r = r_random;
            p.g = g_random;
            p.b = b_random;
            global_instances_cloud_random_color.points.push_back(p);
        }
        global_instances_id_list.push_back(id);
    }

    if (if_visualize) std::cout << "*** Global instances id size: " << global_instances_id_list.size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB> background_for_save;
    if (!background_accumulated.points.empty()) {
        background_accumulated.width = static_cast<uint32_t>(background_accumulated.points.size());
        background_accumulated.height = 1;
        background_accumulated.is_dense = true;
        cloud_generator.voxelFilter(background_accumulated, background_for_save);
        // Glass / specular surfaces produce sparse stray depth in the unlabeled (black) background.
        // Voxel merging alone keeps those outliers; SOR removes points with few nearby neighbors.
        if (!background_for_save.points.empty()) {
            pcl::PointCloud<pcl::PointXYZRGB> background_sor;
            cloud_generator.sorFilter(background_for_save, background_sor, 50, 1.0f);
            background_for_save = std::move(background_sor);
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB> map_with_background;
    map_with_background.points = global_instances_cloud.points;
    for (const auto& pt : background_for_save.points) {
        map_with_background.points.push_back(pt);
    }
    map_with_background.width = static_cast<uint32_t>(map_with_background.points.size());
    map_with_background.height = 1;
    map_with_background.is_dense = true;

    // Show the global instances cloud
    if (if_visualize) {
        pcl::visualization::PCLVisualizer viewer("Global Instances");
        viewer.addPointCloud<pcl::PointXYZRGB>(global_instances_cloud_random_color.makeShared(), "global_instances");
        viewer.spin();
    }

    // Save the global instances cloud (objects only; same as before)
    pcl::io::savePLYFileBinary(output_ply_dir + "instance_cloud.ply", global_instances_cloud);
    pcl::io::savePLYFileBinary(output_ply_dir + "colored_instances.ply", global_instances_cloud_random_color);
    pcl::io::savePLYFileBinary(output_ply_dir + "instance_cloud_with_background.ply", map_with_background);

    return 0;
}
