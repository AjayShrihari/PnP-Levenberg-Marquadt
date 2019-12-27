# PnP-Levenberg-Marquadt
Motion estimation using Iterative PnP resection
We need to recover pose using an iterative Perspective n point algorithm given a reconstructed point cloud scene. The steps needed are as follows:

* Obtain a set of 2D-3D correspondences between the the image and the point cloud. This is easy to obtain since we already have the generated image.

* For this set of correspondences compute the total reprojection error c = i kx i − P k X i k 2 where
P k = K[R k |t k ], X i is the 3D point in the world frame, x i is its corresponding projection.

* Solve for the pose T k that minimizes this non-linear reprojection error using a Gauss-Newton (GN)
scheme. Recall that in GN we start with some initial estimated value x o and iteratively refine the
estimate using x 1 = ∆ x +x 0 , where ∆ x is obtained by solving the normal equations J T J∆ x = −J T e,
until convergence.

* The main steps in this scheme are computing the corresponding Jacobians and updating the es-
timates correctly. For our problem, we use a 12 × 1 vector parameterization for T k (the top 3 × 4
submatrix). We run the optimization for different choices of initialization.




