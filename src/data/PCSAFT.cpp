#include <Eigen/Dense>

namespace teqp::saft::PCSAFT::PCSAFTMatrices{

// From: The OG Gross & Sadowski model; doi: 10.1021/ie0003887
namespace GrossSadowski2001{

extern const auto a = (Eigen::Matrix<double, 3, 7>() <<
           0.9105631445, 0.6361281449, 2.6861347891, -26.547362491, 97.759208784, -159.59154087, 91.297774084,
          -0.3084016918, 0.1860531159, -2.5030047259, 21.419793629, -65.255885330, 83.318680481, -33.746922930,
          -0.0906148351, 0.4527842806, 0.5962700728, -1.7241829131, -4.1302112531, 13.776631870, -8.6728470368).finished();

extern const auto b = (Eigen::Matrix<double, 3, 7>() <<
           0.7240946941, 2.2382791861, -4.0025849485, -21.003576815, 26.855641363, 206.55133841, -355.60235612,
          -0.5755498075, 0.6995095521, 3.8925673390, -17.215471648, 192.67226447, -161.82646165, -165.20769346,
          0.0976883116, -0.2557574982, -9.1558561530, 20.642075974, -38.804430052, 93.626774077, -29.666905585).finished();
}

// From: An Approach to Improve Speed of Sound Calculation within PC-SAFT Framework; doi: 10.1021/ie3018127
namespace LiangIECR2012{

extern const auto a = (Eigen::Matrix<double, 3, 7>() <<
    0.836215101666, 2.201683842453, -11.25210310939, 37.841836899902, -68.035304263396, 69.952369867326, -42.828905226651,
    -0.411727190935913, 2.37426400571265, -21.0620603419144, 105.65718855671,  -298.665894225644, 468.695983731173, -316.673589664169,
    0.0319867672916212, -2.75137756155503, 25.7581175334397, -103.082737163044, 239.569856365622 , -320.622085430506, 186.50494276364).finished();

extern const auto b = (Eigen::Matrix<double, 3, 7>() <<
    0.627209841336118 , 4.02517816132384, -22.6660554011051, 70.0153445172765, -129.548046066679, 150.197401680241, -86.9664928046989,
    -0.622507280536237, 1.86478114256654, -22.5660800172653, 153.069895818654, -321.710866534244, 453.083735030445, -233.232187026907,
    -0.0303061275320169, 2.59554209415371, 11.5803588817289, -145.915305288352, 354.901071174401, -151.675970200232, -282.837925415568).finished();
}

// From: New Variant of the Universal Constants in the Perturbed Chain-Statistical Associating Fluid Theory Equation of State; doi: http://dx.doi.org/10.1021/ie503925h
namespace LiangIECR2014{

extern const auto a = (Eigen::Matrix<double, 3, 7>() <<
    0.961597, 0.414449, 0.689253, -7.43899, 31.8755, -54.8833, 27.3613,
    -0.333416, 0.358440, -0.219088, 0.285586, 5.88256, -22.4931, 21.2109,
    -0.0632483, 0.287134, -0.105309, 7.16607, -37.5008, 68.8002, -40.3177).finished();

extern const auto b = (Eigen::Matrix<double, 3, 7>() <<
    0.548398, 2.07176, -1.84013, -29.9683, 160.445, -206.106, 51.6201,
    -0.277226, 0.0621105, 4.29640, -39.9453, 214.911, -232.563, 60.9734,
    -0.150684, -0.507877, 1.24227, -48.7052, 63.1155, 205.380, -262.437).finished();
}

}
