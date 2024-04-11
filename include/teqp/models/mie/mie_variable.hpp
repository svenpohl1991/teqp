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

        enum combining_rule { ONEFLUID, ONEFLUID_RED, ONEFLUID_RED_LINEAR, ONEFLUID_DENSE, LINEAR, LORENTZ };

        inline auto linear_mixing(const double& x, const double& y) {
            return (x + y) / 2.0;
        }

        class AncTerm {
        public:
            Eigen::ArrayXd n, t;
            double tred, dred;
            std::string type;
        };

        struct fluid {
            double lambdas, epsilons, sigmas, m, I;
            AncTerm anc_dl, anc_dv;
            bool is_sphere;
        };

        // Combining rules for one fluid approximation
        // (1) Simple Van der Waals one fluid combininb rule
        template<typename RHOTYPE, typename MoleFracType>
        inline auto combining_rules_one_fluid(RHOTYPE& rhostar, MoleFracType& molefrac, std::vector<fluid> f, const bool& is_spherical, Eigen::ArrayXXd k_mat_l, Eigen::ArrayXXd k_mat_s, Eigen::ArrayXXd k_mat_e, Eigen::ArrayXd gammat, Eigen::ArrayXd betat, Eigen::ArrayXd gamma_dense, combining_rule mix_rule) {
            auto ncomp = f.size();
            using resulttype = std::common_type_t<decltype(molefrac[0]), decltype(rhostar)>;
            using rhotype = std::common_type_t<decltype(rhostar)>;
            std::vector<std::vector<resulttype>> sigma_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<std::vector<resulttype>> eps_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<std::vector<resulttype>> lambda_ij(ncomp, std::vector<resulttype>(ncomp, 0.0));
            std::vector<resulttype> m_mixed(ncomp, 0.0);
            resulttype m_mix = 0.0;
            resulttype sigma_mean = 0.0;
            resulttype sigma_mean_gamma = 0.0;
            resulttype epsilon_mean = 0.0;
            resulttype lambda_mean = 0.0;

            if (is_spherical) {
                m_mix = 1.0;
            }
            else
            {
                for (auto i = 0; i < ncomp; i++) {
                    m_mix += f[i].m * molefrac[i];
                }
            }

            Eigen::ArrayXd sig_;
            Eigen::ArrayXd lambda_;
            Eigen::ArrayXd epsilon_;

            auto red_func = [molefrac](auto p, auto b, auto g) {
                resulttype val = 0.0;
                for (size_t i = 0; i < molefrac.size(); i++) {
                    val = val + molefrac[i] * molefrac[i] * p(i);
                }
                val = val + 2.0 * molefrac[0] * molefrac[1] * b * g * (molefrac[0] + molefrac[1]) / (b * b * molefrac[0] + molefrac[1]) * sqrt(p(0) * p(1));
                return val;
            };

            auto red_func_d = [molefrac](auto p, auto b, auto g) {
                resulttype val = 0.0;
                for (size_t i = 0; i < molefrac.size(); i++) {
                    val = val + molefrac[i] * molefrac[i] / p(i);
                }
                val = val + 2.0 * molefrac[0] * molefrac[1] * b * g * (molefrac[0] + molefrac[1]) / (b * b * molefrac[0] + molefrac[1]) * 0.125 * pow(1.0 / pow(p(0), 1.0 / 3.0) + 1.0 / pow(p(1), 1.0 / 3.0), 3.0);

                return val;
            };

            // Switch between combing rules for interaction of molecular parameters
            switch (mix_rule) {
            case ONEFLUID:
                if (ncomp > 1) {
                    for (auto i = 0; i < ncomp; i++) {
                        for (auto j = 0; j < ncomp; j++) {
                            lambda_ij[i][j] = (1.0 - k_mat_l(i, j)) * sqrt((f[i].lambdas - 3.0) * (f[j].lambdas - 3.0)) + 3.0;
                            sigma_ij[i][j] = (1.0 - k_mat_s(i, j)) * linear_mixing(f[i].sigmas, f[j].sigmas);
                            eps_ij[i][j] = (1.0 - k_mat_e(i, j)) * sqrt(pow(f[i].sigmas, 3.0) * pow(f[j].sigmas, 3.0)) / pow(sigma_ij[i][j], 3.0) * sqrt(f[i].epsilons * f[j].epsilons);
                        }
                    }
                    // Calculate the mean values for the one fluid approximation
                    for (auto i = 0; i < ncomp; i++) { for (auto j = 0; j < ncomp; j++) { sigma_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0); } }
                    for (auto i = 0; i < ncomp; i++) { for (auto j = 0; j < ncomp; j++) { epsilon_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0) * eps_ij[i][j]; } }
                    for (auto i = 0; i < ncomp; i++) { for (auto j = 0; j < ncomp; j++) { lambda_mean += molefrac[i] * molefrac[j] * f[i].m * f[j].m * pow(sigma_ij[i][j], 3.0) * eps_ij[i][j] * lambda_ij[i][j]; } }

                    lambda_mean = lambda_mean / epsilon_mean;
                    epsilon_mean = epsilon_mean / sigma_mean;
                    sigma_mean = pow(sigma_mean / pow(m_mix, 2.0), 1.0 / 3.0);

                }
                else
                {
                    sigma_mean = f[0].sigmas;
                    epsilon_mean = f[0].epsilons;
                    lambda_mean = f[0].lambdas;
                }
                break;


            case ONEFLUID_RED:
                // so far only for binary mixtures available!
                if (ncomp > 1) {
                    sig_ = (Eigen::ArrayXd(2) << f[0].sigmas, f[1].sigmas).finished();
                    lambda_ = (Eigen::ArrayXd(2) << f[0].lambdas, f[1].lambdas).finished();
                    epsilon_ = (Eigen::ArrayXd(2) << f[0].epsilons, f[1].epsilons).finished();
                    sigma_mean = 1.0 / red_func_d(sig_, betat[0], gammat[0]);
                    epsilon_mean = red_func(epsilon_, betat[1], gammat[1]);
                    lambda_mean = red_func(lambda_, betat[2], gammat[2]);
                }
                else
                {
                    sigma_mean = f[0].sigmas;
                    epsilon_mean = f[0].epsilons;
                    lambda_mean = f[0].lambdas;
                }
                break;
            default:
                break;
            }

            return std::make_tuple(sigma_mean, epsilon_mean, lambda_mean, m_mix);
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

            int nr_fld;
            bool is_sphere;

            Eigen::ArrayXd kij_vec, betat, gammat, gamma_dense;
            Eigen::ArrayXXd k_mat_lambda, k_mat_sigma, k_mat_epsilon;
            combining_rule mix_rule;
            std::map<std::string, combining_rule>  comb_rule = { {"one-fluid",ONEFLUID} , {"one-fluid-red",ONEFLUID_RED} , {"one-fluid-linear",ONEFLUID_RED_LINEAR} };
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
                    fld[i].m = j.at("mie").at(f).at("segment");
                    fld[i].I = j.at("mie").at(f).at("I");
                    i++;
                }

                nr_fld = fluids.size();
                is_sphere = true;

                // check if only sphericals are included
                for (size_t i = 0; i < nr_fld; i++)
                {
                    if (fld[i].m > 1.0) {
                        is_sphere = false;
                    }
                }

                // get mie parameter
                auto spec = j.at("mie").at("parameter");
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


                k_mat_lambda = Eigen::ArrayXXd::Zero(nr_fluids, nr_fluids);
                k_mat_sigma = Eigen::ArrayXXd::Zero(nr_fluids, nr_fluids);
                k_mat_epsilon = Eigen::ArrayXXd::Zero(nr_fluids, nr_fluids);

                // Binary interaction parameter
                if (kij.size() == 0) {
                    // Check if a mixture is present
                    if (nr_fluids > 1) {
                        if (mix_rule == ONEFLUID || mix_rule == ONEFLUID_DENSE) {
                            auto spec_mix = j.at("kij");
                            // Find fluids
                            for (size_t i = 0; i < nr_fluids - 1; i++) {
                                for (size_t k = i + 1; k < nr_fluids; k++) {
                                    // build name combinations
                                    std::string name1 = fluids[i] + "-" + fluids[k];
                                    std::string name2 = fluids[k] + "-" + fluids[i];
                                    if (spec_mix.contains(name1)) {
                                        //kij_vec = toeig(spec_mix.at(name1)).head(3);
                                        auto kmat_loc = spec_mix.at(name1);
                                        k_mat_lambda(i, k) = kmat_loc[0];
                                        k_mat_lambda(k, i) = kmat_loc[0];
                                        k_mat_sigma(i, k) = kmat_loc[1];
                                        k_mat_sigma(k, i) = kmat_loc[1];
                                        k_mat_epsilon(i, k) = kmat_loc[2];
                                        k_mat_epsilon(k, i) = kmat_loc[2];
                                    }
                                    else if (spec_mix.contains(name2)) {
                                        //kij_vec = toeig(spec_mix.at(name2)).head(3);
                                        auto kmat_loc = spec_mix.at(name2);
                                        k_mat_lambda(i, k) = kmat_loc[0];
                                        k_mat_lambda(k, i) = kmat_loc[0];
                                        k_mat_sigma(i, k) = kmat_loc[1];
                                        k_mat_sigma(k, i) = kmat_loc[1];
                                        k_mat_epsilon(i, k) = kmat_loc[2];
                                        k_mat_epsilon(k, i) = kmat_loc[2];
                                    }
                                }
                            }
                        }
                        else if (mix_rule == ONEFLUID_RED) {
                            auto spec_mix = j.at("one_fluid_red");
                            for (size_t i = 0; i < nr_fluids - 1; i++) {
                                for (size_t k = i + 1; k < nr_fluids; k++) {
                                    // build name combinations
                                    std::string name1 = fluids[i] + "-" + fluids[k];
                                    std::string name2 = fluids[k] + "-" + fluids[i];
                                    if (spec_mix.contains(name1)) {
                                        auto parameter = spec_mix.at(name1);
                                        betat = toeig(parameter.at("betat")).head(3);
                                        gammat = toeig(parameter.at("gammat")).head(3);
                                    }
                                    else if (spec_mix.contains(name2)) {
                                        auto parameter = spec_mix.at(name2);
                                        betat = toeig(parameter.at("betat")).head(3);
                                        gammat = toeig(parameter.at("gammat")).head(3);
                                    }
                                }
                            }
                        }
                    }
                    else {
                        kij_vec = (Eigen::ArrayXd(3) << 0.0, 0.0, 0.0).finished();
                    }
                }
                else
                {
                    if (mix_rule == ONEFLUID) {
                        for (size_t i = 0; i < nr_fluids - 1; i++) {
                            for (size_t k = i + 1; k < nr_fluids; k++) {
                                k_mat_lambda(i, k) = kij[0];
                                k_mat_lambda(k, i) = kij[0];
                                k_mat_sigma(i, k) = kij[1];
                                k_mat_sigma(k, i) = kij[1];
                                k_mat_epsilon(i, k) = kij[2];
                                k_mat_epsilon(k, i) = kij[2];
                            }
                        }
                    }
                    else if (mix_rule == ONEFLUID_RED) {
                        betat = (Eigen::ArrayXd(3) << kij[0], kij[1], kij[2]).finished();
                        gammat = (Eigen::ArrayXd(3) << kij[3], kij[4], kij[5]).finished();
                    }
                    else if (mix_rule == ONEFLUID_DENSE) {
                        gamma_dense = (Eigen::ArrayXd(1) << kij[0]).finished();
                    }
                }
            }

            template<typename MoleFracType>
            auto R(const MoleFracType&) const { return NAvo * kBoltz; }


            template<typename ETYPE, typename LTYPE, typename MTYPE>
            inline auto get_tc(const ETYPE& e, const LTYPE& l, const MTYPE& m) const {
                return e * (tc_p[0] + tc_p[1] / l + tc_p[2] / (l * l * l)) *
                    (1.0 + 0.5135 * (m - 1.0) * l - 0.0344 * (m - 1.0) * l * l + 0.0037 * (m - 1.0) * (m - 1.0) * (m - 1.0) * l) /
                    (1.0 + 0.3220 * (m - 1.0) * l - 0.0223 * (m - 1.0) * l * l);
            }

            template<typename STYPE, typename LTYPE, typename MTYPE>
            inline auto get_dc(const STYPE& s, const LTYPE& l, const MTYPE& m) const {
                return 1E3 * m * (dc_p[0] + dc_p[1] * log(l) / log(10.0)) *
                    (1.0 - 0.0681 * (m - 1.0) * l + 0.0029 * (m - 1.0) * pow(l, 2.0)) /
                    (1.0 + 0.0462 * (m - 1.0) * l - 0.0019 * (m - 1.0) * pow(l, 2.0)) * 1E27 / (NAvo * pow(s, 3.0)) / m;
            }


            // Approximate the mixture with the one fluid model
            template<typename TTYPE, typename RHOTYPE, typename MoleFracType>
            auto one_fluid(TTYPE& Tstar, RHOTYPE& rhostar, MoleFracType& molefrac) const {

                using resulttype = std::common_type_t<decltype(Tstar), decltype(molefrac[0]), decltype(rhostar)>;
                resulttype l = 0.0;
                resulttype s = 0.0;
                resulttype e = 0.0;
                resulttype m = 0.0;

                std::tie(s, e, l, m) = combining_rules_one_fluid(rhostar, molefrac, fld, is_sphere, k_mat_lambda, k_mat_sigma, k_mat_epsilon, gammat, betat, gamma_dense, mix_rule);

                resulttype tc = get_tc(e, l, m);
                resulttype dc = get_dc(s, l, m);

                resulttype tau = tc / Tstar;
                resulttype delta = rhostar / dc;

                // Calculate coefficients
                std::vector< resulttype> n_pol;
                std::vector< resulttype> n_exp;
                std::vector< resulttype> n_gbs;
                for (size_t i = 0; i < t_pol.size(); i++) { n_pol.push_back(c1_pol[i] + c2_pol[i] / l); }
                for (size_t i = 0; i < t_exp.size(); i++) { n_exp.push_back(c1_exp[i] + c2_exp[i] / l); }
                for (size_t i = 0; i < t_gbs.size(); i++) { n_gbs.push_back(c1_gbs[i] + c2_gbs[i] / l); }

                std::vector< resulttype> pol, exp_, gbs;
                for (size_t i = 0; i < t_pol.size(); i++) { pol.push_back(n_pol[i] * pow(tau, t_pol[i]) * pow(delta, d_pol[i])); }
                for (size_t i = 0; i < t_exp.size(); i++) { exp_.push_back(n_exp[i] * pow(tau, t_exp[i]) * pow(delta, d_exp[i]) * exp(-pow(delta, p[i]))); }
                for (size_t i = 0; i < t_gbs.size(); i++) { gbs.push_back(n_gbs[i] * pow(tau, t_gbs[i]) * pow(delta, d_gbs[i]) * exp(-eta[i] * (delta - eps[i]) * (delta - eps[i]) - beta[i] * (tau - gam[i]) * (tau - gam[i]))); }

                auto alpha_r_sphere = std::reduce(pol.begin(), pol.end()) + std::reduce(exp_.begin(), exp_.end()) + std::reduce(gbs.begin(), gbs.end());
                resulttype alpha_r_all = 0.0;
                alpha_r_all = alpha_r_sphere;
                return alpha_r_all;
            }

            // Input is temperature in K, density in mol/m^3 and molefractions
            template<typename TTYPE, typename RHOTYPE, typename MoleFracType>
            auto alphar(const TTYPE& Tstar, const RHOTYPE& rhostar, const MoleFracType& molefrac) const {
                using resulttype = std::common_type_t<decltype(Tstar), decltype(molefrac[0]), decltype(rhostar)>;
                resulttype alpha_r_all = 0.0;
                alpha_r_all = one_fluid(Tstar, rhostar, molefrac);
                return forceeval(alpha_r_all);
            }
        };
    }
}

