#include "teqp/models/multifluid.hpp"
#include <Eigen/Dense>
#include <math.h>
#include "teqp/models/cubics.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

namespace teqp {
    namespace Mie {

        enum combining_rule { ONEFLUID, ONEFLUID_RED, ONEFLUID_RED_LINEAR, ONEFLUID_DENSE, LINEAR, LORENTZ, STRUCTURE_BASED };

        inline auto linear_mixing(const double& x, const double& y) {
            return (x + y) / 2.0;
        }

        template<typename LTYPE>
        inline auto l_to_m(const LTYPE& L_star) {
            return 1. + 0.2177 * L_star + 3.1498 * L_star * L_star - 3.6738 * L_star * L_star * L_star + 1.3063 * L_star * L_star * L_star * L_star;
        }

        struct fluid {
            double lambdas, epsilons, sigmas, m, I, L_star;
            bool is_sphere;
        };

        // Combining rules for one fluid approximation
        // (1) Simple Van der Waals one fluid combininb rule
        template<typename RHOTYPE, typename MoleFracType>
        inline auto combining_rules_one_fluid(RHOTYPE& rhostar, MoleFracType& molefrac, std::vector<fluid> f, const bool& is_spherical, Eigen::ArrayXXd k_mat_l, Eigen::ArrayXXd k_mat_s, Eigen::ArrayXXd k_mat_e, Eigen::ArrayXd gammat, Eigen::ArrayXd betat, Eigen::ArrayXd gamma_dense, combining_rule mix_rule) {
            auto ncomp = f.size();
            using resulttype = std::common_type_t<decltype(molefrac[0])>;
            std::vector<std::vector<resulttype>> sigma_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<std::vector<resulttype>> eps_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<std::vector<resulttype>> lambda_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<resulttype> m_mixed(ncomp, 0.0);
            resulttype m_mix = 0.0;
            resulttype L_mix = 0.0;
            resulttype sigma_mean = 0.0;
            resulttype sigma_mean_gamma = 0.0;
            resulttype epsilon_mean = 0.0;
            resulttype lambda_mean = 0.0;


            for (auto i = 0; i < ncomp; i++) { m_mix += f[i].m * molefrac[i]; }
            for (auto i = 0; i < ncomp; i++) { L_mix += f[i].L_star * molefrac[i]; }


            // Switch between combing rules for interaction of molecular parameters
            resulttype factor_i = 1.0;
            resulttype factor_j = 1.0;
            switch (mix_rule) {
            case ONEFLUID:
            case STRUCTURE_BASED:
                for (auto i = 0; i < ncomp; i++) {
                    for (auto j = 0; j < ncomp; j++) {
                        lambda_ij[i][j] = (1.0 - k_mat_l(i, j)) * sqrt((f[i].lambdas - 3.0) * (f[j].lambdas - 3.0)) + 3.0;
                        sigma_ij[i][j] = (1.0 - k_mat_s(i, j)) * linear_mixing(f[i].sigmas, f[j].sigmas);
                        eps_ij[i][j] = (1.0 - k_mat_e(i, j)) * sqrt(pow(f[i].sigmas, 3.0) * pow(f[j].sigmas, 3.0)) / pow(sigma_ij[i][j], 3.0) * sqrt(f[i].epsilons * f[j].epsilons) / factor_i;
                    }
                }
                // Calculate the mean values for the one fluid approximation
                for (auto i = 0; i < ncomp; i++) { for (auto j = 0; j < ncomp; j++) { sigma_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0); } }
                for (auto i = 0; i < ncomp; i++) { for (auto j = 0; j < ncomp; j++) { epsilon_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0) * eps_ij[i][j]; } }
                for (auto i = 0; i < ncomp; i++) { for (auto j = 0; j < ncomp; j++) { lambda_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0) * eps_ij[i][j] * lambda_ij[i][j]; } }

                lambda_mean = lambda_mean / epsilon_mean;
                epsilon_mean = epsilon_mean / sigma_mean;
                sigma_mean = pow(sigma_mean / pow(m_mix, 2.0), 1.0 / 3.0);
                break;

            default:
                break;
            }

            return std::make_tuple(sigma_mean, epsilon_mean, lambda_mean, m_mix, L_mix);
        }

        // Combining rules for one fluid approximation
// (1) Simple Van der Waals one fluid combininb rule
        template<typename RHOTYPE, typename MoleFracType>
        inline auto combining_rules_structure_fluid(RHOTYPE& rhostar, MoleFracType& molefrac, std::vector<fluid> f, const bool& is_spherical, Eigen::ArrayXXd k_mat_l, Eigen::ArrayXXd k_mat_s, Eigen::ArrayXXd k_mat_e, Eigen::ArrayXd gammat, Eigen::ArrayXd betat, Eigen::ArrayXd gamma_dense, combining_rule mix_rule) {
            auto ncomp = f.size();
            using resulttype = std::common_type_t<decltype(molefrac[0])>;
            std::vector<std::vector<resulttype>> sigma_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<std::vector<resulttype>> eps_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<std::vector<resulttype>> lambda_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<resulttype> m_mixed(ncomp, 0.0);
            resulttype m_mix = 0.0;
            resulttype L_mix = 0.0;
            resulttype m_mean = 0.0;
            resulttype sigma_mean = 0.0;
            resulttype sigma_mean_gamma = 0.0;
            resulttype epsilon_mean = 0.0;
            resulttype lambda_mean = 0.0;


            for (auto i = 0; i < ncomp; i++) { m_mix += f[i].m * molefrac[i]; }
            for (auto i = 0; i < ncomp; i++) { L_mix += f[i].L_star * molefrac[i]; }


            // Switch between combing rules for interaction of molecular parameters
            resulttype factor_i = 1.0;
            resulttype factor_j = 1.0;
            switch (mix_rule) {
            case ONEFLUID:
            case STRUCTURE_BASED:
                for (auto i = 0; i < ncomp - 1; i++) {
                    for (auto j = i + 1; j < ncomp; j++) {
                        lambda_ij[i][j] = (1.0 - k_mat_l(i, j)) * sqrt((f[i].lambdas - 3.0) * (f[j].lambdas - 3.0)) + 3.0;
                        sigma_ij[i][j] = (1.0 - k_mat_s(i, j)) * linear_mixing(f[i].sigmas, f[j].sigmas);
                        if (f[i].L_star > 0.0 && f[j].L_star == 0.0) {
                            factor_i = 2.0;
                        }
                        else if (f[j].L_star > 0.0 && f[i].L_star == 0.0) {
                            factor_i = 2.0;
                        }
                        else if (f[i].L_star == 0.0 && f[j].L_star == 0.0) {
                            factor_i = 4.0;
                        }
                        else if (f[j].L_star == 0.0 && f[i].L_star == 0.0) {
                            factor_i = 4.0;
                        }
                        else if (f[j].L_star > 0.0 && f[i].L_star > 0.0) {
                            factor_i = 1.0;
                        }
                        else if (f[i].L_star > 0.0 && f[j].L_star > 0.0) {
                            factor_i = 1.0;
                        }
                        eps_ij[i][j] = (1.0 - k_mat_e(i, j)) * sqrt(pow(f[i].sigmas, 3.0) * pow(f[j].sigmas, 3.0)) / pow(sigma_ij[i][j], 3.0) * sqrt(f[i].epsilons * f[j].epsilons) / factor_i;
                    }
                }
                // Calculate the mean values for the one fluid approximation
                for (auto i = 0; i < ncomp - 1; i++) { for (auto j = i + 1; j < ncomp; j++) { m_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m; } }
                for (auto i = 0; i < ncomp - 1; i++) { for (auto j = i + 1; j < ncomp; j++) { sigma_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0); } }
                for (auto i = 0; i < ncomp - 1; i++) { for (auto j = i + 1; j < ncomp; j++) { epsilon_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0) * eps_ij[i][j]; } }
                for (auto i = 0; i < ncomp - 1; i++) { for (auto j = i + 1; j < ncomp; j++) { lambda_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0) * eps_ij[i][j] * lambda_ij[i][j]; } }

                lambda_mean = lambda_mean / epsilon_mean;
                epsilon_mean = epsilon_mean / sigma_mean;
                sigma_mean = pow(sigma_mean / m_mean, 1.0 / 3.0); // pow(sigma_mean / pow(m_mix, 2.0), 1.0 / 3.0);
                break;

            default:
                break;
            }

            return std::make_tuple(sigma_mean, epsilon_mean, lambda_mean, m_mix, L_mix);
        }

        template<typename Model>
        auto get_vle_pure_start(const Model& model, double& T, const int idx) {
            return get_sat_dense(T, model, idx);
        }

        const  double kBoltz = 1.380649E-23;
        const double NAvo = 6.02214076E23;

        class MieElong {

        private:

            const  double kBoltz = 1.380649E-23;
            const double NAvo = 6.02214076E23;
            using EArray6 = Eigen::Array<double, 6, 1>;
            using EArray4 = Eigen::Array<double, 4, 1>;


            Eigen::ArrayXd c1_pol;
            Eigen::ArrayXd c1_exp;
            Eigen::ArrayXd c1_gbs;
            Eigen::ArrayXd c2_pol;
            Eigen::ArrayXd c2_exp;
            Eigen::ArrayXd c2_gbs;
            Eigen::ArrayXd t_pol;
            Eigen::ArrayXd t_exp;
            Eigen::ArrayXd t_gbs;
            Eigen::ArrayXd d_pol;
            Eigen::ArrayXd d_exp;
            Eigen::ArrayXd d_gbs;
            Eigen::ArrayXd p;
            Eigen::ArrayXd eta;
            Eigen::ArrayXd beta;
            Eigen::ArrayXd gam;
            Eigen::ArrayXd eps;
            Eigen::ArrayXd tc_p;
            Eigen::ArrayXd dc_p;
            Eigen::ArrayXd alpha_p_pol;
            Eigen::ArrayXd alpha_p_exp;
            Eigen::ArrayXd alpha_p_gbs;

            Eigen::ArrayXd d1_pol;
            Eigen::ArrayXd d2_pol;
            Eigen::ArrayXd d3_pol;
            Eigen::ArrayXd d1_exp;
            Eigen::ArrayXd d2_exp;
            Eigen::ArrayXd d3_exp;
            Eigen::ArrayXd d1_gbs;
            Eigen::ArrayXd d2_gbs;
            Eigen::ArrayXd d3_gbs;
            Eigen::ArrayXd t_pol_m;
            Eigen::ArrayXd t_exp_m;
            Eigen::ArrayXd t_gbs_m;
            Eigen::ArrayXd d_pol_m;
            Eigen::ArrayXd d_exp_m;
            Eigen::ArrayXd d_gbs_m;
            Eigen::ArrayXd o_pol;
            Eigen::ArrayXd o_exp;
            Eigen::ArrayXd o_gbs;
            Eigen::ArrayXd p_m;
            Eigen::ArrayXd eta_m;
            Eigen::ArrayXd beta_m;
            Eigen::ArrayXd gam_m;
            Eigen::ArrayXd eps_m;

            int nr_fld;
            bool is_sphere;

            Eigen::ArrayXd kij_vec, betat, gammat, gamma_dense;
            Eigen::ArrayXXd k_mat_lambda, k_mat_sigma, k_mat_epsilon, k_mat_l;
            combining_rule mix_rule;
            std::vector<double> L_stars;
            std::map<std::string, combining_rule>  comb_rule = { {"one-fluid",ONEFLUID} , {"one-fluid-red",ONEFLUID_RED} , {"one-fluid-linear",ONEFLUID_RED_LINEAR} , {"structure",STRUCTURE_BASED} };
        public:

            std::vector<fluid> fld;

            MieElong(const std::string& path, const std::vector<std::string>& fluids, const std::string& combining_rule_in, const std::vector<double>& kij = {}) {

                std::string filepath = std::filesystem::is_regular_file(path) ? path : path;
                nlohmann::json j = load_a_JSON_file(filepath);
                std::string combining_rule = combining_rule_in;
                auto nr_fluids = fluids.size();
                std::transform(combining_rule.begin(), combining_rule.end(), combining_rule.begin(), [](unsigned char c) { return std::tolower(c); });


                try {
                    // Attempt to access the map with the user's input  
                    int value = comb_rule.at(combining_rule);
                    mix_rule = comb_rule[combining_rule];
                }
                catch (const std::out_of_range& e) {
                    // Handle the exception when the key is not found
                    std::cerr << "Error: " << e.what() << " is not a valid mixing rule." << std::endl;
                }

                fld.resize(fluids.size());
                int i = 0;
                for (auto f : fluids)
                {
                    fld[i].lambdas = j.at("mie").at(f).at("lambda");
                    fld[i].epsilons = j.at("mie").at(f).at("epsilon");
                    fld[i].sigmas = j.at("mie").at(f).at("sigma");
                    fld[i].L_star = j.at("mie").at(f).at("L");
                    fld[i].m = l_to_m(fld[i].L_star);
                    L_stars.push_back(fld[i].L_star);
                    fld[i].I = j.at("mie").at(f).at("I");
                    i++;
                }

                nr_fld = fluids.size();
                is_sphere = true;

                // check if only sphericals are included
                for (size_t i = 0; i < nr_fld; i++)
                {
                    if (fld[i].L_star > 0.0) {
                        is_sphere = false;
                    }
                }

                auto spec = j.at("mie").at("parameter");

                // #######################
                // Read mie parameter for spherical part
                // #######################
                auto n_pol = static_cast<int>(spec.at("c1_pol").size());
                auto n_exp = static_cast<int>(spec.at("c1_exp").size());
                auto n_gbs = static_cast<int>(spec.at("c1_gbs").size());
                c1_pol = toeig(spec.at("c1_pol")).head(n_pol);
                c1_exp = toeig(spec.at("c1_exp")).head(n_exp);
                c1_gbs = toeig(spec.at("c1_gbs")).head(n_gbs);
                c2_pol = toeig(spec.at("c2_pol")).head(n_pol);
                c2_exp = toeig(spec.at("c2_exp")).head(n_exp);
                c2_gbs = toeig(spec.at("c2_gbs")).head(n_gbs);
                t_pol = toeig(spec.at("t_pol")).head(n_pol);
                t_exp = toeig(spec.at("t_exp")).head(n_exp);
                t_gbs = toeig(spec.at("t_gbs")).head(n_gbs);
                d_pol = toeig(spec.at("d_pol")).head(n_pol);
                d_exp = toeig(spec.at("d_exp")).head(n_exp);
                d_gbs = toeig(spec.at("d_gbs")).head(n_gbs);
                p = toeig(spec.at("p")).head(n_exp);
                eta = toeig(spec.at("eta")).head(n_gbs);
                beta = toeig(spec.at("beta")).head(n_gbs);
                gam = toeig(spec.at("gam")).head(n_gbs);
                eps = toeig(spec.at("eps")).head(n_gbs);

                auto n_tc = static_cast<int>(spec.at("tc_p").size());
                auto n_dc = static_cast<int>(spec.at("dc_p").size());
                tc_p = toeig(spec.at("tc_p")).head(n_tc);
                dc_p = toeig(spec.at("dc_p")).head(n_dc);


                // #######################
                // Read mie parameter for elongated part
                // #######################
                auto n_pol_m = static_cast<int>(spec.at("d1_pol").size());
                auto n_exp_m = static_cast<int>(spec.at("d1_exp").size());
                auto n_gbs_m = static_cast<int>(spec.at("d1_gbs").size());
                auto n_alpha = static_cast<int>(spec.at("alpha_p_pol").size());
                d1_pol = toeig(spec.at("d1_pol")).head(n_pol_m);
                d2_pol = toeig(spec.at("d2_pol")).head(n_pol_m);
                d3_pol = toeig(spec.at("d3_pol")).head(n_pol_m);
                d1_exp = toeig(spec.at("d1_exp")).head(n_exp_m);
                d2_exp = toeig(spec.at("d2_exp")).head(n_exp_m);
                d3_exp = toeig(spec.at("d3_exp")).head(n_exp_m);
                d1_gbs = toeig(spec.at("d1_gbs")).head(n_gbs_m);
                d2_gbs = toeig(spec.at("d2_gbs")).head(n_gbs_m);
                d3_gbs = toeig(spec.at("d3_gbs")).head(n_gbs_m);
                t_pol_m = toeig(spec.at("t_pol_m")).head(n_pol_m);
                t_exp_m = toeig(spec.at("t_exp_m")).head(n_exp_m);
                t_gbs_m = toeig(spec.at("t_gbs_m")).head(n_gbs_m);
                d_pol_m = toeig(spec.at("d_pol_m")).head(n_pol_m);
                d_exp_m = toeig(spec.at("d_exp_m")).head(n_exp_m);
                d_gbs_m = toeig(spec.at("d_gbs_m")).head(n_gbs_m);

                p_m = toeig(spec.at("p_m")).head(n_exp_m);
                eta_m = toeig(spec.at("eta_m")).head(n_gbs_m);
                beta_m = toeig(spec.at("beta_m")).head(n_gbs_m);
                gam_m = toeig(spec.at("gam_m")).head(n_gbs_m);
                eps_m = toeig(spec.at("eps_m")).head(n_gbs_m);
                alpha_p_pol = toeig(spec.at("alpha_p_pol")).head(n_alpha);
                alpha_p_exp = toeig(spec.at("alpha_p_exp")).head(n_alpha);
                alpha_p_gbs = toeig(spec.at("alpha_p_gbs")).head(n_alpha);


                k_mat_lambda = Eigen::ArrayXXd::Zero(nr_fluids, nr_fluids);
                k_mat_sigma = Eigen::ArrayXXd::Zero(nr_fluids, nr_fluids);
                k_mat_epsilon = Eigen::ArrayXXd::Zero(nr_fluids, nr_fluids);
                k_mat_l = Eigen::ArrayXXd::Zero(nr_fluids, nr_fluids);

                if (nr_fluids > 1) {
                    auto spec_mix = j.at("kij");
                    if (mix_rule == STRUCTURE_BASED) {
                        auto spec_mix = j.at("kij");
                        // Find fluids
                        for (size_t i = 0; i < nr_fluids - 1; i++) {
                            for (size_t k = i + 1; k < nr_fluids; k++) {
                                // build name combinations
                                std::string name1 = fluids[i] + "-" + fluids[k];
                                std::string name2 = fluids[k] + "-" + fluids[i];
                                if (spec_mix.contains(name1)) {
                                    auto kmat_loc = spec_mix.at(name1);
                                    k_mat_lambda(i, k) = kmat_loc[0];
                                    k_mat_lambda(k, i) = kmat_loc[0];
                                    k_mat_sigma(i, k) = kmat_loc[1];
                                    k_mat_sigma(k, i) = kmat_loc[1];
                                    k_mat_epsilon(i, k) = kmat_loc[2];
                                    k_mat_epsilon(k, i) = kmat_loc[2];
                                    k_mat_l(i, k) = kmat_loc[3];
                                    k_mat_l(k, i) = kmat_loc[3];
                                }
                                else if (spec_mix.contains(name2)) {
                                    auto kmat_loc = spec_mix.at(name2);
                                    k_mat_lambda(i, k) = kmat_loc[0];
                                    k_mat_lambda(k, i) = kmat_loc[0];
                                    k_mat_sigma(i, k) = kmat_loc[1];
                                    k_mat_sigma(k, i) = kmat_loc[1];
                                    k_mat_epsilon(i, k) = kmat_loc[2];
                                    k_mat_epsilon(k, i) = kmat_loc[2];
                                    k_mat_l(i, k) = kmat_loc[3];
                                    k_mat_l(k, i) = kmat_loc[3];
                                }
                            }
                        }
                    }
                }
            }

            template<typename MoleFracType>
            auto R(const MoleFracType&) const { return NAvo * kBoltz; }


            template<typename ETYPE, typename LTYPE, typename MTYPE>
            inline auto get_tc(const ETYPE& epsilon, const LTYPE& lambda, const MTYPE& L_star) const {
                MTYPE ms = l_to_m(L_star);
                MTYPE ms_min1 = ms - 1.0;
                return  forceeval(epsilon * (tc_p[0] + tc_p[1] / lambda + tc_p[2] / (lambda * lambda * lambda)) *
                    (1.0 + tc_p[3] * ms_min1 / lambda + tc_p[4] * ms_min1 * ms_min1 / lambda + tc_p[5] * ms_min1 * ms_min1 * ms_min1 / lambda)
                    / (1.0 + tc_p[6] * ms_min1 / lambda + tc_p[7] * ms_min1 * ms_min1 / lambda + tc_p[8] * ms_min1 * ms_min1 * ms_min1 / lambda));
            }

            template<typename STYPE, typename LTYPE, typename MTYPE>
            inline auto get_dc(const STYPE& s, const LTYPE& lambda, const MTYPE& L_star) const {
                MTYPE ms = l_to_m(L_star);
                MTYPE ms_min1 = ms - 1.0;
                return forceeval(1E3 * (dc_p[0] + dc_p[1] * log(lambda) / log(10.0)) * (1.0 + dc_p[2] * ms_min1) / (1.0 + dc_p[3] * ms_min1)
                    * 1E27 / (NAvo * s * s * s));
            }

            template<typename LTYPE, typename MTYPE, typename ALPHATYPE>
            inline auto get_alpha(const LTYPE& lambda, const MTYPE& L_star, const ALPHATYPE& alpha) const {
                MTYPE ms = l_to_m(L_star);
                MTYPE ms_min1 = ms - 1.0;
                return alpha[0] + alpha[1] * ms_min1 + alpha[2] * lambda + alpha[3] * powi(ms_min1, 2) + alpha[4] * ms_min1 * lambda + alpha[5] * powi(ms_min1, 3) + alpha[6] * powi(ms_min1, 2) * lambda;
            }

            template<typename DELTATYPE, typename TAUTYPE, typename LAMBDATYPE>
            inline auto get_spherical_contribution(const DELTATYPE& delta, const TAUTYPE& tau, const LAMBDATYPE& lambda) const {
                using resulttype = std::common_type_t<decltype(delta), decltype(tau)>;
                // Calculate coefficients for spherical part
                std::vector<resulttype> n_pol(t_pol.size()), n_exp(t_exp.size()), n_gbs(t_gbs.size());

                for (size_t i = 0; i < t_pol.size(); i++) {
                    n_pol[i] = c1_pol[i] + c2_pol[i] / lambda;
                }
                for (size_t i = 0; i < t_exp.size(); i++) {
                    n_exp[i] = c1_exp[i] + c2_exp[i] / lambda;
                }
                for (size_t i = 0; i < t_gbs.size(); i++) {
                    n_gbs[i] = c1_gbs[i] + c2_gbs[i] / lambda;
                }

                std::vector<resulttype> pol(t_pol.size()), exp_(t_exp.size()), gbs(t_gbs.size());
                for (size_t i = 0; i < t_pol.size(); i++) {
                    pol[i] = n_pol[i] * pow(tau, t_pol[i]) * pow(delta, d_pol[i]);
                }
                for (size_t i = 0; i < t_exp.size(); i++) {
                    exp_[i] = n_exp[i] * pow(tau, t_exp[i]) * pow(delta, d_exp[i]) * exp(-pow(delta, p[i]));
                }
                for (size_t i = 0; i < t_gbs.size(); i++) {
                    gbs[i] = n_gbs[i] * pow(tau, t_gbs[i]) * pow(delta, d_gbs[i]) * exp(-eta[i] * (delta - eps[i]) * (delta - eps[i]) - beta[i] * (tau - gam[i]) * (tau - gam[i]));
                }

                return std::reduce(pol.begin(), pol.end()) +
                    std::reduce(exp_.begin(), exp_.end()) +
                    std::reduce(gbs.begin(), gbs.end());
            }

            template<typename DELTATYPE, typename TAUTYPE, typename ALPHATYPE>
            inline auto get_elongated_contribution(const DELTATYPE& delta, const TAUTYPE& tau, const ALPHATYPE& alpha_pol, const ALPHATYPE& alpha_exp, const ALPHATYPE& alpha_gbs) const {
                using resulttype = std::common_type_t<decltype(delta), decltype(tau)>;
                // Calculate coefficients for elongated part
                std::vector<resulttype> n_pol_m(t_pol_m.size()), n_exp_m(t_exp_m.size()), n_gbs_m(t_gbs_m.size());

                for (size_t i = 0; i < t_pol_m.size(); i++) {
                    n_pol_m[i] = d1_pol[i] + d2_pol[i] * alpha_pol + d3_pol[i] * alpha_pol * alpha_pol;
                }
                for (size_t i = 0; i < t_exp_m.size(); i++) {
                    n_exp_m[i] = d1_exp[i] + d2_exp[i] * alpha_exp + d3_exp[i] * alpha_exp * alpha_exp;
                }
                for (size_t i = 0; i < t_gbs_m.size(); i++) {
                    n_gbs_m[i] = d1_gbs[i] + d2_gbs[i] * alpha_gbs + d3_gbs[i] * alpha_gbs * alpha_gbs;
                }

                std::vector<resulttype> pol_m(t_pol_m.size()), exp_m(t_exp_m.size()), gbs_m(t_gbs_m.size());
                for (size_t i = 0; i < t_pol_m.size(); i++) {
                    pol_m[i] = n_pol_m[i] * pow(tau, t_pol_m[i]) * pow(delta, d_pol_m[i]);
                }
                for (size_t i = 0; i < t_exp_m.size(); i++) {
                    exp_m[i] = n_exp_m[i] * pow(tau, t_exp_m[i]) * pow(delta, d_exp_m[i]) * exp(-pow(delta, p_m[i]));
                }
                for (size_t i = 0; i < t_gbs_m.size(); i++) {
                    gbs_m[i] = n_gbs_m[i] * pow(tau, t_gbs_m[i]) * pow(delta, d_gbs_m[i]) * exp(-eta_m[i] * (delta - eps_m[i]) * (delta - eps_m[i]) - beta_m[i] * (tau - gam_m[i]) * (tau - gam_m[i]));
                }

                return std::reduce(pol_m.begin(), pol_m.end()) +
                    std::reduce(exp_m.begin(), exp_m.end()) +
                    std::reduce(gbs_m.begin(), gbs_m.end());
            }

            template<typename TTYPE, typename RHOTYPE, typename MoleFracType, typename EPSTYPE, typename SIGTYPE, typename  LAMBTYPE, typename LTYPE>
            auto get_alpha_r(TTYPE& Tstar, RHOTYPE& rhostar, MoleFracType& molefrac, EPSTYPE& epsilon, SIGTYPE& sigma, LAMBTYPE& lambda, LTYPE& L_star) const {
                using resulttype = std::common_type_t<decltype(Tstar), decltype(molefrac[0]), decltype(rhostar)>;
                resulttype tc = get_tc(epsilon, lambda, L_star);
                resulttype dc = get_dc(sigma, lambda, L_star);
                resulttype tau = tc / Tstar;
                resulttype delta = rhostar / dc;
                resulttype alpha_r_elong = 0.0;
                resulttype alpha_r_sphere = 0.0;
                resulttype alpha_pol = 0.0;
                resulttype alpha_exp = 0.0;
                resulttype alpha_gbs = 0.0;
                resulttype segment = l_to_m(L_star);
                alpha_r_sphere = get_spherical_contribution(delta, tau, lambda);
                if (abs(segment - 1.0) > 1E-14) {
                    alpha_pol = get_alpha(lambda, L_star, alpha_p_pol);
                    alpha_exp = get_alpha(lambda, L_star, alpha_p_exp);
                    alpha_gbs = get_alpha(lambda, L_star, alpha_p_gbs);
                    alpha_r_elong = get_elongated_contribution(delta, tau, alpha_pol, alpha_exp, alpha_gbs);
                }

                resulttype alpha_r_all = 0.0;

                if (abs(segment - 1.0) < 1E-14) {
                    alpha_r_all = segment * alpha_r_sphere;
                }
                else {
                    alpha_r_all = segment * alpha_r_sphere + (segment - 1.0) * alpha_r_elong;
                }

                return forceeval(alpha_r_all);
            }

            // // Approximate the mixture with the one fluid model
            // template<typename TTYPE, typename RHOTYPE, typename MoleFracType>
            // auto one_fluid(TTYPE& Tstar, RHOTYPE& rhostar, MoleFracType& molefrac) const {

                // using resulttype = std::common_type_t<decltype(Tstar), decltype(molefrac[0]), decltype(rhostar)>;
                // resulttype lambda = 0.0;
                // resulttype sigma = 0.0;
                // resulttype epsilon = 0.0;
                // resulttype L_star = 0.0;
                // resulttype m = 0.0;
                // std::tie(sigma, epsilon, lambda, m, L_star) = combining_rules_one_fluid(rhostar, molefrac, fld, is_sphere, k_mat_lambda, k_mat_sigma, k_mat_epsilon, gammat, betat, gamma_dense, mix_rule);
                // return get_alpha_r(Tstar, rhostar, molefrac, epsilon, sigma, lambda, L_star, m);
            // }


            template<typename TTYPE, typename RHOTYPE, typename MoleFracType>
            auto contribution_fluid(TTYPE& Tstar, RHOTYPE& rhostar, MoleFracType& molefrac) const {
                using resulttype = std::common_type_t<decltype(Tstar), decltype(molefrac[0]), decltype(rhostar)>;
                resulttype lambda = 0.0;
                resulttype sigma = 0.0;
                resulttype epsilon = 0.0;
                resulttype elongation = 0.0;
                std::vector<resulttype> alpha_comb;
                double factor = 1.0;
                int ncomp = fld.size();

                for (size_t i = 0; i < ncomp; i++) {
                    for (size_t j = 0; j < ncomp; j++) {
                        elongation = (1.0 - k_mat_l(i,j)) * linear_mixing(fld[i].L_star, fld[j].L_star);
                        lambda =     (1.0 - k_mat_lambda(i, j)) * sqrt((fld[i].lambdas - 3.0) * (fld[j].lambdas - 3.0)) + 3.0;
                        sigma =      (1.0 - k_mat_sigma(i, j)) * pow(0.5 * (pow(fld[i].sigmas, 3.0) + pow(fld[j].sigmas, 3.0)), 1.0 / 3.0); //linear_mixing(fld[i].sigmas, fld[j].sigmas);
                        epsilon =    (1.0 - k_mat_epsilon(i, j)) * sqrt(pow(fld[i].sigmas, 3.0) * pow(fld[j].sigmas, 3.0)) / pow(sigma, 3.0) * sqrt(fld[i].epsilons * fld[j].epsilons) / factor;
                        alpha_comb.push_back(molefrac[i] * molefrac[j] * get_alpha_r(Tstar, rhostar, molefrac, epsilon, sigma, lambda, elongation));
                    }
                }

                return std::reduce(alpha_comb.begin(), alpha_comb.end());
            }


            // Input is temperature in K, density in mol/m^3 and molefractions
            template<typename TTYPE, typename RHOTYPE, typename MoleFracType>
            auto alphar(const TTYPE& Tstar, const RHOTYPE& rhostar, const MoleFracType& molefrac) const {
                using resulttype = std::common_type_t<decltype(Tstar), decltype(molefrac[0]), decltype(rhostar)>;
                resulttype alpha_r_all = 0.0;
                // if (mix_rule == ONEFLUID) { alpha_r_all = one_fluid(Tstar, rhostar, molefrac); }
                if (mix_rule == STRUCTURE_BASED) { alpha_r_all = contribution_fluid(Tstar, rhostar, molefrac); }

                return forceeval(alpha_r_all);
            }
        };
    }
}
