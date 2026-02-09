/*
 * C++ Extensions for Robot Movement AI
 * =====================================
 * 
 * Extensiones C++ optimizadas para operaciones críticas:
 * - Cinemática inversa rápida
 * - Optimización de trayectorias
 * - Operaciones matriciales
 * - Detección de colisiones
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace py = pybind11;
using namespace Eigen;

// ============================================================================
// Fast Matrix Operations
// ============================================================================

class FastMatrixOps {
public:
    // Multiplicación de matrices optimizada
    static py::array_t<double> matmul(
        py::array_t<double> a,
        py::array_t<double> b
    ) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        
        if (buf_a.ndim != 2 || buf_b.ndim != 2) {
            throw std::runtime_error("Input arrays must be 2D");
        }
        
        if (buf_a.shape[1] != buf_b.shape[0]) {
            throw std::runtime_error("Matrix dimensions incompatible");
        }
        
        const Map<const MatrixXd> mat_a((double*)buf_a.ptr, buf_a.shape[0], buf_a.shape[1]);
        const Map<const MatrixXd> mat_b((double*)buf_b.ptr, buf_b.shape[0], buf_b.shape[1]);
        
        MatrixXd result = mat_a * mat_b;
        
        auto result_array = py::array_t<double>({result.rows(), result.cols()});
        auto result_buf = result_array.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        std::memcpy(result_ptr, result.data(), result.size() * sizeof(double));
        
        return result_array;
    }
    
    // Inversa de matriz usando descomposición LU
    static py::array_t<double> inv(py::array_t<double> a) {
        auto buf = a.request();
        
        if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
            throw std::runtime_error("Input must be a square matrix");
        }
        
        const Map<const MatrixXd> mat((double*)buf.ptr, buf.shape[0], buf.shape[1]);
        MatrixXd inv_mat = mat.partialPivLu().inverse();
        
        auto result = py::array_t<double>({inv_mat.rows(), inv_mat.cols()});
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        std::memcpy(result_ptr, inv_mat.data(), inv_mat.size() * sizeof(double));
        
        return result;
    }
    
    // Determinante
    static double det(py::array_t<double> a) {
        auto buf = a.request();
        
        if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
            throw std::runtime_error("Input must be a square matrix");
        }
        
        Map<MatrixXd> mat((double*)buf.ptr, buf.shape[0], buf.shape[1]);
        return mat.determinant();
    }
    
    // Transpuesta
    static py::array_t<double> transpose(py::array_t<double> a) {
        auto buf = a.request();
        
        if (buf.ndim != 2) {
            throw std::runtime_error("Input must be 2D");
        }
        
        const Map<const MatrixXd> mat((double*)buf.ptr, buf.shape[0], buf.shape[1]);
        MatrixXd result = mat.transpose();
        
        auto result_array = py::array_t<double>({result.rows(), result.cols()});
        auto result_buf = result_array.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        std::memcpy(result_ptr, result.data(), result.size() * sizeof(double));
        return result_array;
    }
    
    // Norma de matriz (Frobenius)
    static double norm(py::array_t<double> a) {
        auto buf = a.request();
        
        if (buf.ndim != 2) {
            throw std::runtime_error("Input must be 2D");
        }
        
        const Map<const MatrixXd> mat((double*)buf.ptr, buf.shape[0], buf.shape[1]);
        return mat.norm();
    }
    
    // Traza
    static double trace(py::array_t<double> a) {
        auto buf = a.request();
        
        if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
            throw std::runtime_error("Input must be a square matrix");
        }
        
        const Map<const MatrixXd> mat((double*)buf.ptr, buf.shape[0], buf.shape[1]);
        return mat.trace();
    }
};

// ============================================================================
// Fast Inverse Kinematics
// ============================================================================

class FastIK {
private:
    std::vector<double> link_lengths;
    std::vector<std::pair<double, double>> joint_limits;
    int max_iterations;
    double tolerance;
    
public:
    FastIK(
        std::vector<double> lengths,
        std::vector<std::pair<double, double>> limits,
        int max_iter = 100,
        double tol = 1e-6
    ) : link_lengths(lengths), joint_limits(limits), 
        max_iterations(max_iter), tolerance(tol) {}
    
    Vector3d forward_kinematics(const std::vector<double>& angles) const {
        double x = 0.0, y = 0.0, z = 0.0;
        double theta = 0.0;
        const size_t n = std::min(angles.size(), link_lengths.size());
        
        for (size_t i = 0; i < n; ++i) {
            theta += angles[i];
            const double cos_theta = std::cos(theta);
            const double sin_theta = std::sin(theta);
            const double len = link_lengths[i];
            x += len * cos_theta;
            y += len * sin_theta;
            if (i % 2 != 0) {
                z += len * std::sin(angles[i]);
            }
        }
        
        return Vector3d(x, y, z);
    }
    
    std::vector<double> solve(
        const Vector3d& target_pos,
        const std::vector<double>& initial_guess
    ) {
        std::vector<double> angles = initial_guess;
        if (angles.size() != link_lengths.size()) {
            angles.resize(link_lengths.size(), 0.0);
        }
        
        const double eps = 1e-6;
        const double learning_rate = 0.1;
        const double eps_sq = tolerance * tolerance;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            Vector3d current_pos = forward_kinematics(angles);
            Vector3d error = target_pos - current_pos;
            const double error_sq = error.squaredNorm();
            
            if (error_sq < eps_sq) {
                break;
            }
            
            for (size_t i = 0; i < angles.size(); ++i) {
                std::vector<double> angles_perturbed = angles;
                angles_perturbed[i] += eps;
                
                Vector3d pos_perturbed = forward_kinematics(angles_perturbed);
                Vector3d jacobian_col = (pos_perturbed - current_pos) / eps;
                
                const double jacobian_norm_sq = jacobian_col.squaredNorm();
                if (jacobian_norm_sq > 1e-20) {
                    double step = error.dot(jacobian_col) / jacobian_norm_sq;
                    angles[i] += step * learning_rate;
                    
                    if (i < joint_limits.size()) {
                        angles[i] = std::max(joint_limits[i].first, 
                                            std::min(joint_limits[i].second, angles[i]));
                    }
                }
            }
        }
        
        return angles;
    }
};

// ============================================================================
// Fast Trajectory Optimization
// ============================================================================

class FastTrajectoryOptimizer {
private:
    double energy_weight;
    double time_weight;
    double smoothness_weight;
    
public:
    FastTrajectoryOptimizer(
        double e_weight = 0.3,
        double t_weight = 0.3,
        double s_weight = 0.2
    ) : energy_weight(e_weight), time_weight(t_weight), 
        smoothness_weight(s_weight) {}
    
    // Optimizar trayectoria usando gradiente descendente
    py::array_t<double> optimize(
        py::array_t<double> trajectory,
        py::array_t<double> obstacles
    ) {
        auto traj_buf = trajectory.request();
        auto obs_buf = obstacles.request();
        
        if (traj_buf.ndim != 2 || traj_buf.shape[1] != 3) {
            throw std::runtime_error("Trajectory must be Nx3 array");
        }
        
        int n_points = traj_buf.shape[0];
        auto result = py::array_t<double>(traj_buf.shape);
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        double* traj_ptr = (double*)traj_buf.ptr;
        
        std::memcpy(result_ptr, traj_ptr, n_points * 3 * sizeof(double));
        
        const int n_iter = 10;
        const double w_prev = 0.25;
        const double w_curr = 0.5;
        const double w_next = 0.25;
        
        for (int iter = 0; iter < n_iter; ++iter) {
            for (int i = 1; i < n_points - 1; ++i) {
                const int idx_base = i * 3;
                const int idx_prev = (i - 1) * 3;
                const int idx_next = (i + 1) * 3;
                
                result_ptr[idx_base] = w_prev * result_ptr[idx_prev] + 
                                      w_curr * result_ptr[idx_base] + 
                                      w_next * result_ptr[idx_next];
                result_ptr[idx_base + 1] = w_prev * result_ptr[idx_prev + 1] + 
                                          w_curr * result_ptr[idx_base + 1] + 
                                          w_next * result_ptr[idx_next + 1];
                result_ptr[idx_base + 2] = w_prev * result_ptr[idx_prev + 2] + 
                                          w_curr * result_ptr[idx_base + 2] + 
                                          w_next * result_ptr[idx_next + 2];
            }
        }
        
        return result;
    }
};

// ============================================================================
// Fast Collision Detection
// ============================================================================

class FastCollisionDetector {
public:
    static bool point_sphere_collision(
        const Vector3d& point,
        const Vector3d& sphere_center,
        double radius
    ) {
        const double radius_sq = radius * radius;
        return (point - sphere_center).squaredNorm() < radius_sq;
    }
    
    // Detección de colisión punto-caja
    static bool point_box_collision(
        const Vector3d& point,
        const Vector3d& box_min,
        const Vector3d& box_max
    ) {
        return point.x() >= box_min.x() && point.x() <= box_max.x() &&
               point.y() >= box_min.y() && point.y() <= box_max.y() &&
               point.z() >= box_min.z() && point.z() <= box_max.z();
    }
    
    // Verificar colisión de trayectoria con obstáculos
    static bool trajectory_collision(
        py::array_t<double> trajectory,
        py::array_t<double> obstacles
    ) {
        auto traj_buf = trajectory.request();
        auto obs_buf = obstacles.request();
        
        if (traj_buf.ndim != 2 || traj_buf.shape[1] != 3) {
            return false;
        }
        
        double* traj_ptr = (double*)traj_buf.ptr;
        double* obs_ptr = (double*)obs_buf.ptr;
        
        const int n_points = traj_buf.shape[0];
        const int n_obstacles = obs_buf.shape[0];
        
        for (int i = 0; i < n_points; ++i) {
            const int idx = i * 3;
            const Vector3d point(traj_ptr[idx], traj_ptr[idx+1], traj_ptr[idx+2]);
            
            for (int j = 0; j < n_obstacles; ++j) {
                const int obs_idx = j * 4;
                const Vector3d center(obs_ptr[obs_idx], obs_ptr[obs_idx+1], obs_ptr[obs_idx+2]);
                const double radius = obs_ptr[obs_idx+3];
                
                if (point_sphere_collision(point, center, radius)) {
                    return true;
                }
            }
        }
        
        return false;
    }
};

// ============================================================================
// Fast 3D Transformations
// ============================================================================

class FastTransform3D {
public:
    // Rotación alrededor del eje Z
    static py::array_t<double> rotation_z(double angle) {
        auto result = py::array_t<double>({3, 3});
        auto buf = result.request();
        double* ptr = (double*)buf.ptr;
        
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        
        ptr[0] = c;  ptr[1] = -s; ptr[2] = 0;
        ptr[3] = s;  ptr[4] = c;  ptr[5] = 0;
        ptr[6] = 0;  ptr[7] = 0;  ptr[8] = 1;
        
        return result;
    }
    
    // Rotación alrededor del eje Y
    static py::array_t<double> rotation_y(double angle) {
        auto result = py::array_t<double>({3, 3});
        auto buf = result.request();
        double* ptr = (double*)buf.ptr;
        
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        
        ptr[0] = c;  ptr[1] = 0;  ptr[2] = s;
        ptr[3] = 0;  ptr[4] = 1;  ptr[5] = 0;
        ptr[6] = -s; ptr[7] = 0;  ptr[8] = c;
        
        return result;
    }
    
    // Rotación alrededor del eje X
    static py::array_t<double> rotation_x(double angle) {
        auto result = py::array_t<double>({3, 3});
        auto buf = result.request();
        double* ptr = (double*)buf.ptr;
        
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        
        ptr[0] = 1;  ptr[1] = 0;  ptr[2] = 0;
        ptr[3] = 0;  ptr[4] = c;  ptr[5] = -s;
        ptr[6] = 0;  ptr[7] = s;  ptr[8] = c;
        
        return result;
    }
    
    // Aplicar rotación a punto
    static py::array_t<double> rotate_point(
        py::array_t<double> rotation_matrix,
        py::array_t<double> point
    ) {
        auto rot_buf = rotation_matrix.request();
        auto pt_buf = point.request();
        
        if (rot_buf.shape[0] != 3 || rot_buf.shape[1] != 3) {
            throw std::runtime_error("Rotation matrix must be 3x3");
        }
        if (pt_buf.size != 3) {
            throw std::runtime_error("Point must be 3D");
        }
        
        const Map<const Matrix3d> rot((double*)rot_buf.ptr);
        const Map<const Vector3d> pt((double*)pt_buf.ptr);
        Vector3d result = rot * pt;
        
        auto result_array = py::array_t<double>(3);
        auto result_buf = result_array.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        std::memcpy(result_ptr, result.data(), 3 * sizeof(double));
        return result_array;
    }
};

// ============================================================================
// Fast Vector Operations
// ============================================================================

class FastVectorOps {
public:
    // Normalizar vector
    static py::array_t<double> normalize(py::array_t<double> vec) {
        auto buf = vec.request();
        
        if (buf.ndim != 1) {
            throw std::runtime_error("Vector must be 1D");
        }
        
        double* vec_ptr = (double*)buf.ptr;
        double norm_sq = 0.0;
        
        for (py::ssize_t i = 0; i < buf.size; ++i) {
            norm_sq += vec_ptr[i] * vec_ptr[i];
        }
        
        const double norm = std::sqrt(norm_sq);
        if (norm < 1e-10) {
            throw std::runtime_error("Cannot normalize zero vector");
        }
        
        auto result = py::array_t<double>(buf.size);
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        for (py::ssize_t i = 0; i < buf.size; ++i) {
            result_ptr[i] = vec_ptr[i] / norm;
        }
        
        return result;
    }
    
    // Distancia euclidiana
    static double distance(
        py::array_t<double> a,
        py::array_t<double> b
    ) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        
        if (buf_a.size != buf_b.size) {
            throw std::runtime_error("Vectors must have same size");
        }
        
        double* a_ptr = (double*)buf_a.ptr;
        double* b_ptr = (double*)buf_b.ptr;
        double dist_sq = 0.0;
        
        for (py::ssize_t i = 0; i < buf_a.size; ++i) {
            const double diff = a_ptr[i] - b_ptr[i];
            dist_sq += diff * diff;
        }
        
        return std::sqrt(dist_sq);
    }
    
    // Producto punto
    static double dot(py::array_t<double> a, py::array_t<double> b) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        
        if (buf_a.size != buf_b.size) {
            throw std::runtime_error("Vectors must have same size");
        }
        
        double* a_ptr = (double*)buf_a.ptr;
        double* b_ptr = (double*)buf_b.ptr;
        double dot = 0.0;
        
        for (py::ssize_t i = 0; i < buf_a.size; ++i) {
            dot += a_ptr[i] * b_ptr[i];
        }
        
        return dot;
    }
    
    // Producto cruz 3D
    static py::array_t<double> cross(
        py::array_t<double> a,
        py::array_t<double> b
    ) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        
        if (buf_a.size != 3 || buf_b.size != 3) {
            throw std::runtime_error("Vectors must be 3D");
        }
        
        double* a_ptr = (double*)buf_a.ptr;
        double* b_ptr = (double*)buf_b.ptr;
        
        auto result = py::array_t<double>(3);
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        result_ptr[0] = a_ptr[1] * b_ptr[2] - a_ptr[2] * b_ptr[1];
        result_ptr[1] = a_ptr[2] * b_ptr[0] - a_ptr[0] * b_ptr[2];
        result_ptr[2] = a_ptr[0] * b_ptr[1] - a_ptr[1] * b_ptr[0];
        
        return result;
    }
};

// ============================================================================
// Fast Interpolation
// ============================================================================

class FastInterpolation {
public:
    // Interpolación lineal
    static py::array_t<double> linear(
        py::array_t<double> points,
        int num_output
    ) {
        auto buf = points.request();
        
        if (buf.ndim != 2 || buf.shape[1] != 3) {
            throw std::runtime_error("Points must be Nx3");
        }
        
        const int n_input = buf.shape[0];
        if (n_input < 2) {
            throw std::runtime_error("Need at least 2 points");
        }
        
        auto result = py::array_t<double>({num_output, 3});
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        double* points_ptr = (double*)buf.ptr;
        
        for (int i = 0; i < num_output; ++i) {
            const double t = static_cast<double>(i) / (num_output - 1);
            const double segment_t = t * (n_input - 1);
            int segment = static_cast<int>(segment_t);
            double local_t = segment_t - segment;
            
            if (segment >= n_input - 1) {
                segment = n_input - 2;
                local_t = 1.0;
            }
            
            const int seg_idx = segment * 3;
            const int next_idx = (segment + 1) * 3;
            const int out_idx = i * 3;
            
            for (int j = 0; j < 3; ++j) {
                const double p0 = points_ptr[seg_idx + j];
                const double p1 = points_ptr[next_idx + j];
                result_ptr[out_idx + j] = p0 + local_t * (p1 - p0);
            }
        }
        
        return result;
    }
};

// ============================================================================
// Fast Quaternion Operations
// ============================================================================

class FastQuaternion {
public:
    // Crear quaternion desde ángulo y eje
    static py::array_t<double> from_axis_angle(
        py::array_t<double> axis,
        double angle
    ) {
        auto axis_buf = axis.request();
        if (axis_buf.size != 3) {
            throw std::runtime_error("Axis must be 3D");
        }
        
        double* axis_ptr = (double*)axis_buf.ptr;
        Vector3d axis_vec(axis_ptr[0], axis_ptr[1], axis_ptr[2]);
        axis_vec.normalize();
        
        const double half_angle = angle / 2.0;
        const double s = std::sin(half_angle);
        
        auto result = py::array_t<double>(4);
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        result_ptr[0] = std::cos(half_angle);  // w
        result_ptr[1] = axis_vec[0] * s;       // x
        result_ptr[2] = axis_vec[1] * s;       // y
        result_ptr[3] = axis_vec[2] * s;       // z
        
        return result;
    }
    
    // Multiplicar quaternions
    static py::array_t<double> multiply(
        py::array_t<double> q1,
        py::array_t<double> q2
    ) {
        auto q1_buf = q1.request();
        auto q2_buf = q2.request();
        
        if (q1_buf.size != 4 || q2_buf.size != 4) {
            throw std::runtime_error("Quaternions must be 4D");
        }
        
        double* q1_ptr = (double*)q1_buf.ptr;
        double* q2_ptr = (double*)q2_buf.ptr;
        
        auto result = py::array_t<double>(4);
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        // q1 * q2
        result_ptr[0] = q1_ptr[0] * q2_ptr[0] - q1_ptr[1] * q2_ptr[1] - 
                         q1_ptr[2] * q2_ptr[2] - q1_ptr[3] * q2_ptr[3];
        result_ptr[1] = q1_ptr[0] * q2_ptr[1] + q1_ptr[1] * q2_ptr[0] + 
                         q1_ptr[2] * q2_ptr[3] - q1_ptr[3] * q2_ptr[2];
        result_ptr[2] = q1_ptr[0] * q2_ptr[2] - q1_ptr[1] * q2_ptr[3] + 
                         q1_ptr[2] * q2_ptr[0] + q1_ptr[3] * q2_ptr[1];
        result_ptr[3] = q1_ptr[0] * q2_ptr[3] + q1_ptr[1] * q2_ptr[2] - 
                         q1_ptr[2] * q2_ptr[1] + q1_ptr[3] * q2_ptr[0];
        
        return result;
    }
    
    // Convertir quaternion a matriz de rotación
    static py::array_t<double> to_rotation_matrix(py::array_t<double> q) {
        auto q_buf = q.request();
        if (q_buf.size != 4) {
            throw std::runtime_error("Quaternion must be 4D");
        }
        
        double* q_ptr = (double*)q_buf.ptr;
        const double w = q_ptr[0], x = q_ptr[1], y = q_ptr[2], z = q_ptr[3];
        
        auto result = py::array_t<double>({3, 3});
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        result_ptr[0] = 1 - 2*(y*y + z*z);
        result_ptr[1] = 2*(x*y - w*z);
        result_ptr[2] = 2*(x*z + w*y);
        result_ptr[3] = 2*(x*y + w*z);
        result_ptr[4] = 1 - 2*(x*x + z*z);
        result_ptr[5] = 2*(y*z - w*x);
        result_ptr[6] = 2*(x*z - w*y);
        result_ptr[7] = 2*(y*z + w*x);
        result_ptr[8] = 1 - 2*(x*x + y*y);
        
        return result;
    }
    
    // Normalizar quaternion
    static py::array_t<double> normalize(py::array_t<double> q) {
        auto q_buf = q.request();
        if (q_buf.size != 4) {
            throw std::runtime_error("Quaternion must be 4D");
        }
        
        double* q_ptr = (double*)q_buf.ptr;
        double norm_sq = q_ptr[0]*q_ptr[0] + q_ptr[1]*q_ptr[1] + 
                        q_ptr[2]*q_ptr[2] + q_ptr[3]*q_ptr[3];
        
        if (norm_sq < 1e-10) {
            throw std::runtime_error("Cannot normalize zero quaternion");
        }
        
        const double norm = std::sqrt(norm_sq);
        auto result = py::array_t<double>(4);
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        for (int i = 0; i < 4; ++i) {
            result_ptr[i] = q_ptr[i] / norm;
        }
        
        return result;
    }
};

// ============================================================================
// Fast Homogeneous Transformations
// ============================================================================

class FastHomogeneousTransform {
public:
    // Crear transformación homogénea desde rotación y traslación
    static py::array_t<double> from_rotation_translation(
        py::array_t<double> rotation,
        py::array_t<double> translation
    ) {
        auto rot_buf = rotation.request();
        auto trans_buf = translation.request();
        
        if (rot_buf.shape[0] != 3 || rot_buf.shape[1] != 3) {
            throw std::runtime_error("Rotation must be 3x3");
        }
        if (trans_buf.size != 3) {
            throw std::runtime_error("Translation must be 3D");
        }
        
        auto result = py::array_t<double>({4, 4});
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        double* rot_ptr = (double*)rot_buf.ptr;
        double* trans_ptr = (double*)trans_buf.ptr;
        
        // Matriz de rotación
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result_ptr[i * 4 + j] = rot_ptr[i * 3 + j];
            }
        }
        
        // Traslación
        result_ptr[0 * 4 + 3] = trans_ptr[0];
        result_ptr[1 * 4 + 3] = trans_ptr[1];
        result_ptr[2 * 4 + 3] = trans_ptr[2];
        
        // Última fila
        result_ptr[3 * 4 + 0] = 0.0;
        result_ptr[3 * 4 + 1] = 0.0;
        result_ptr[3 * 4 + 2] = 0.0;
        result_ptr[3 * 4 + 3] = 1.0;
        
        return result;
    }
    
    // Aplicar transformación homogénea a punto
    static py::array_t<double> transform_point(
        py::array_t<double> transform,
        py::array_t<double> point
    ) {
        auto tf_buf = transform.request();
        auto pt_buf = point.request();
        
        if (tf_buf.shape[0] != 4 || tf_buf.shape[1] != 4) {
            throw std::runtime_error("Transform must be 4x4");
        }
        if (pt_buf.size != 3) {
            throw std::runtime_error("Point must be 3D");
        }
        
        double* tf_ptr = (double*)tf_buf.ptr;
        double* pt_ptr = (double*)pt_buf.ptr;
        
        auto result = py::array_t<double>(3);
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        for (int i = 0; i < 3; ++i) {
            result_ptr[i] = tf_ptr[i * 4 + 0] * pt_ptr[0] +
                           tf_ptr[i * 4 + 1] * pt_ptr[1] +
                           tf_ptr[i * 4 + 2] * pt_ptr[2] +
                           tf_ptr[i * 4 + 3];
        }
        
        return result;
    }
    
    // Inversa de transformación homogénea
    static py::array_t<double> inverse(py::array_t<double> transform) {
        auto tf_buf = transform.request();
        if (tf_buf.shape[0] != 4 || tf_buf.shape[1] != 4) {
            throw std::runtime_error("Transform must be 4x4");
        }
        
        double* tf_ptr = (double*)tf_buf.ptr;
        
        auto result = py::array_t<double>({4, 4});
        auto result_buf = result.request();
        double* result_ptr = (double*)result_buf.ptr;
        
        // Transponer rotación
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result_ptr[i * 4 + j] = tf_ptr[j * 4 + i];
            }
        }
        
        // Invertir traslación
        double trans[3] = {
            tf_ptr[0 * 4 + 3],
            tf_ptr[1 * 4 + 3],
            tf_ptr[2 * 4 + 3]
        };
        
        result_ptr[0 * 4 + 3] = -(result_ptr[0 * 4 + 0] * trans[0] +
                                  result_ptr[0 * 4 + 1] * trans[1] +
                                  result_ptr[0 * 4 + 2] * trans[2]);
        result_ptr[1 * 4 + 3] = -(result_ptr[1 * 4 + 0] * trans[0] +
                                  result_ptr[1 * 4 + 1] * trans[1] +
                                  result_ptr[1 * 4 + 2] * trans[2]);
        result_ptr[2 * 4 + 3] = -(result_ptr[2 * 4 + 0] * trans[0] +
                                  result_ptr[2 * 4 + 1] * trans[1] +
                                  result_ptr[2 * 4 + 2] * trans[2]);
        
        // Última fila
        result_ptr[3 * 4 + 0] = 0.0;
        result_ptr[3 * 4 + 1] = 0.0;
        result_ptr[3 * 4 + 2] = 0.0;
        result_ptr[3 * 4 + 3] = 1.0;
        
        return result;
    }
};

// ============================================================================
// Fast Geometry Operations
// ============================================================================

class FastGeometry {
public:
    // Distancia punto a línea
    static double point_to_line_distance(
        py::array_t<double> point,
        py::array_t<double> line_start,
        py::array_t<double> line_end
    ) {
        auto pt_buf = point.request();
        auto start_buf = line_start.request();
        auto end_buf = line_end.request();
        
        if (pt_buf.size != 3 || start_buf.size != 3 || end_buf.size != 3) {
            throw std::runtime_error("Points must be 3D");
        }
        
        double* pt_ptr = (double*)pt_buf.ptr;
        double* start_ptr = (double*)start_buf.ptr;
        double* end_ptr = (double*)end_buf.ptr;
        
        Vector3d p(pt_ptr[0], pt_ptr[1], pt_ptr[2]);
        Vector3d a(start_ptr[0], start_ptr[1], start_ptr[2]);
        Vector3d b(end_ptr[0], end_ptr[1], end_ptr[2]);
        
        Vector3d ab = b - a;
        Vector3d ap = p - a;
        
        double ab_norm_sq = ab.squaredNorm();
        if (ab_norm_sq < 1e-10) {
            return (p - a).norm();
        }
        
        double t = ap.dot(ab) / ab_norm_sq;
        t = std::max(0.0, std::min(1.0, t));
        
        Vector3d closest = a + t * ab;
        return (p - closest).norm();
    }
    
    // Área de triángulo
    static double triangle_area(
        py::array_t<double> a,
        py::array_t<double> b,
        py::array_t<double> c
    ) {
        auto a_buf = a.request();
        auto b_buf = b.request();
        auto c_buf = c.request();
        
        if (a_buf.size != 3 || b_buf.size != 3 || c_buf.size != 3) {
            throw std::runtime_error("Points must be 3D");
        }
        
        double* a_ptr = (double*)a_buf.ptr;
        double* b_ptr = (double*)b_buf.ptr;
        double* c_ptr = (double*)c_buf.ptr;
        
        Vector3d va(a_ptr[0], a_ptr[1], a_ptr[2]);
        Vector3d vb(b_ptr[0], b_ptr[1], b_ptr[2]);
        Vector3d vc(c_ptr[0], c_ptr[1], c_ptr[2]);
        
        Vector3d ab = vb - va;
        Vector3d ac = vc - va;
        Vector3d cross = ab.cross(ac);
        
        return 0.5 * cross.norm();
    }
};

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(cpp_extensions, m) {
    m.doc() = "C++ extensions for Robot Movement AI";
    
    // Fast Matrix Operations
    py::class_<FastMatrixOps>(m, "FastMatrixOps")
        .def_static("matmul", &FastMatrixOps::matmul, "Matrix multiplication")
        .def_static("inv", &FastMatrixOps::inv, "Matrix inverse")
        .def_static("det", &FastMatrixOps::det, "Matrix determinant")
        .def_static("transpose", &FastMatrixOps::transpose, "Matrix transpose")
        .def_static("norm", &FastMatrixOps::norm, "Matrix Frobenius norm")
        .def_static("trace", &FastMatrixOps::trace, "Matrix trace");
    
    // Fast Inverse Kinematics
    py::class_<FastIK>(m, "FastIK")
        .def(py::init<
            std::vector<double>,
            std::vector<std::pair<double, double>>,
            int,
            double
        >())
        .def("solve", &FastIK::solve, "Solve inverse kinematics")
        .def("forward_kinematics", &FastIK::forward_kinematics, "Forward kinematics");
    
    // Fast Trajectory Optimizer
    py::class_<FastTrajectoryOptimizer>(m, "FastTrajectoryOptimizer")
        .def(py::init<double, double, double>())
        .def("optimize", &FastTrajectoryOptimizer::optimize, "Optimize trajectory");
    
    // Fast Collision Detector
    py::class_<FastCollisionDetector>(m, "FastCollisionDetector")
        .def_static("trajectory_collision", &FastCollisionDetector::trajectory_collision,
                   "Check trajectory collision with obstacles")
        .def_static("point_sphere_collision", &FastCollisionDetector::point_sphere_collision,
                   "Check point-sphere collision")
        .def_static("point_box_collision", &FastCollisionDetector::point_box_collision,
                   "Check point-box collision");
    
    // Fast 3D Transformations
    py::class_<FastTransform3D>(m, "FastTransform3D")
        .def_static("rotation_x", &FastTransform3D::rotation_x, "Rotation matrix around X axis")
        .def_static("rotation_y", &FastTransform3D::rotation_y, "Rotation matrix around Y axis")
        .def_static("rotation_z", &FastTransform3D::rotation_z, "Rotation matrix around Z axis")
        .def_static("rotate_point", &FastTransform3D::rotate_point, "Apply rotation to point");
    
    // Fast Vector Operations
    py::class_<FastVectorOps>(m, "FastVectorOps")
        .def_static("normalize", &FastVectorOps::normalize, "Normalize vector")
        .def_static("distance", &FastVectorOps::distance, "Euclidean distance")
        .def_static("dot", &FastVectorOps::dot, "Dot product")
        .def_static("cross", &FastVectorOps::cross, "Cross product (3D)");
    
    // Fast Interpolation
    py::class_<FastInterpolation>(m, "FastInterpolation")
        .def_static("linear", &FastInterpolation::linear, "Linear interpolation");
    
    // Fast Quaternion Operations
    py::class_<FastQuaternion>(m, "FastQuaternion")
        .def_static("from_axis_angle", &FastQuaternion::from_axis_angle,
                   "Create quaternion from axis and angle")
        .def_static("multiply", &FastQuaternion::multiply,
                   "Multiply quaternions")
        .def_static("to_rotation_matrix", &FastQuaternion::to_rotation_matrix,
                   "Convert quaternion to rotation matrix")
        .def_static("normalize", &FastQuaternion::normalize,
                   "Normalize quaternion");
    
    // Fast Homogeneous Transformations
    py::class_<FastHomogeneousTransform>(m, "FastHomogeneousTransform")
        .def_static("from_rotation_translation", 
                   &FastHomogeneousTransform::from_rotation_translation,
                   "Create homogeneous transform from rotation and translation")
        .def_static("transform_point",
                   &FastHomogeneousTransform::transform_point,
                   "Transform point using homogeneous transform")
        .def_static("inverse", &FastHomogeneousTransform::inverse,
                   "Inverse of homogeneous transform");
    
    // Fast Geometry Operations
    py::class_<FastGeometry>(m, "FastGeometry")
        .def_static("point_to_line_distance",
                   &FastGeometry::point_to_line_distance,
                   "Distance from point to line segment")
        .def_static("triangle_area", &FastGeometry::triangle_area,
                   "Area of triangle");
    
    // Standalone functions for convenience
    m.def("fast_inverse_kinematics", [](std::vector<double> lengths,
                                        std::vector<std::pair<double, double>> limits,
                                        Vector3d target,
                                        std::vector<double> initial) {
        FastIK ik(lengths, limits);
        return ik.solve(target, initial);
    });
    
    m.def("fast_trajectory_optimization", [](py::array_t<double> traj,
                                            py::array_t<double> obs) {
        FastTrajectoryOptimizer opt;
        return opt.optimize(traj, obs);
    });
    
    m.def("fast_matrix_operations", [](py::array_t<double> a, py::array_t<double> b) {
        return FastMatrixOps::matmul(a, b);
    });
    
    m.def("fast_collision_detection", [](py::array_t<double> traj,
                                        py::array_t<double> obs) {
        return FastCollisionDetector::trajectory_collision(traj, obs);
    });
}

