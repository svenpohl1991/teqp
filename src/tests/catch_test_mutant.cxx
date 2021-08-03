#include "catch/catch.hpp"

#include "teqp/models/multifluid.hpp"

TEST_CASE("Test construction of mutant", "[mutant]")
{

	std::string coolprop_root = "../mycp";
	auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";

	auto model = build_multifluid_model({ "Nitrogen", "Ethane" }, coolprop_root, BIPcollection);

    std::string s0 = R"({"0": {} })";
    nlohmann::json j0 = nlohmann::json::parse(s0);

    std::string s = R"({
        "0": {
            "1": {
                "BIP": {
                    "betaT": 1.1,
                    "gammaT": 0.9,
                    "betaV": 1.05,
                    "gammaV": 1.3,
                    "Fij": 1.0
                },
                "departure":{
                    "type": "none"
                }
            }
        }
    })";
	nlohmann::json j = nlohmann::json::parse(s);
    auto mutant = build_multifluid_mutant(model, j);

    double T = 300, rho = 300;
    Eigen::ArrayXd molefrac(2); molefrac = 0.5;
    auto Ar02base = TDXDerivatives<decltype(model)>::get_Ar02(model, T, rho, molefrac);
    auto Ar02mut = TDXDerivatives<decltype(mutant)>::get_Ar02(mutant, T, rho, molefrac);
    CHECK(Ar02base != Ar02mut);

}