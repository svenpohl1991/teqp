#include "catch/catch.hpp"

#include "teqp/models/multifluid.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/filesystem.hpp"

using namespace teqp;

TEST_CASE("Test infinite dilution critical locus derivatives for multifluid", "[crit]")
{
    std::string root = "../mycp";

    const auto model = build_multifluid_model({ "Nitrogen", "Ethane" }, root);
    using ct = CriticalTracing<decltype(model), double, Eigen::ArrayXd>;

    for (int i = 0; i < 2; ++i) {
        auto rhoc0 = 1/model.redfunc.vc[i];
        double T0 = model.redfunc.Tc[i];
        Eigen::ArrayXd rhovec0(2); rhovec0.setZero(); rhovec0[i] = rhoc0;

        // Values for infinite dilution
        auto infdil = ct::get_drhovec_dT_crit(model, T0, rhovec0);
        auto epinfdil = ct::eigen_problem(model, T0, rhovec0);

        // Just slightly not infinite dilution, values should be very similar
        Eigen::ArrayXd rhovec0almost = rhovec0; rhovec0almost[1 - i] = 1e-6;
        auto dil = ct::get_drhovec_dT_crit(model, T0, rhovec0almost);
        auto epdil = ct::eigen_problem(model, T0, rhovec0almost);
        int rr = 0;

    }
}

TEST_CASE("Test infinite dilution critical locus derivatives for multifluid with both orders", "[crit]")
{
    std::string root = "../mycp";

    auto pure_endpoint = [&](const std::vector < std::string> &fluids, int i) {
        const auto model = build_multifluid_model(fluids, root);
        using ct = CriticalTracing<decltype(model), double, Eigen::ArrayXd>;
        auto rhoc0 = 1 / model.redfunc.vc[i];
        double T0 = model.redfunc.Tc[i];
        Eigen::ArrayXd rhovec0(2); rhovec0.setZero(); rhovec0[i] = rhoc0;
        // Values for infinite dilution
        auto infdil = ct::get_drhovec_dT_crit(model, T0, rhovec0);
        auto epinfdil = ct::eigen_problem(model, T0, rhovec0);
        auto der = ct::get_derivs(model, T0, rhovec0);
        using tdx = TDXDerivatives<decltype(model), double, Eigen::ArrayXd>;
        auto z = (rhovec0 / rhovec0.sum()).eval();
        auto alphar = model.alphar(T0, rhoc0, z);
        return std::make_tuple(T0, rhoc0, alphar, infdil, epinfdil, der);
    };

    auto [T0, rho0, alphar0, infdil0, eig0, der0] = pure_endpoint({ "Nitrogen", "Ethane" }, 0);
    auto [T1, rho1, alphar1, infdil1, eig1, der1] = pure_endpoint({ "Ethane", "Nitrogen" }, 1);
    CHECK(T0 == T1);
    CHECK(rho0 == rho1);
    CHECK(alphar0 == alphar1);
    CHECK(infdil0(1) == infdil1(0));
    CHECK(infdil0(0) == infdil1(1));

    auto [Ta, rhoa, alphara, infdila, eiga, dera] = pure_endpoint({ "Ethane", "Nitrogen" }, 0);
    auto [Tb, rhob, alpharb, infdilb, eigb, derb] = pure_endpoint({ "Nitrogen", "Ethane" }, 1);
    CHECK(Ta == Tb);
    CHECK(rhoa == rhob);
    CHECK(alphara == alpharb);
    CHECK(infdila(1) == infdilb(0));
    CHECK(infdila(0) == infdilb(1));

    int rr = 0;
}


TEST_CASE("Confirm failure for missing files","[multifluid]") {
    CHECK_THROWS(build_multifluid_model({ "BADFLUID" }, "IMPOSSIBLE PATH", "IMPOSSIBLE PATH.json"));
    CHECK_THROWS(build_multifluid_model({ "BADFLUID" }, "IMPOSSIBLE PATH", "../mycp/dev/mixtures/mixture_binary_pairs.json"));
    CHECK_THROWS(build_multifluid_model({ "Ethane" }, "IMPOSSIBLE PATH"));
}

TEST_CASE("Trace critical locus for nitrogen + ethane", "[crit],[multifluid]")
{
    std::string root = "../mycp";
    const auto model = build_multifluid_model({ "Nitrogen", "Ethane" }, root);

    for (auto ifluid = 0; ifluid < 2; ++ifluid) {
        double T0 = model.redfunc.Tc[ifluid];
        Eigen::ArrayXd rhovec0(2); rhovec0 = 0.0; rhovec0[ifluid] = 1.0 / model.redfunc.vc[ifluid];

        auto tic0 = std::chrono::steady_clock::now();
        std::string filename = "";
        using ct = CriticalTracing<decltype(model), double, Eigen::ArrayXd>;
        TCABOptions opt; opt.init_dt = 100; opt.integration_order = 1;
        auto j = ct::trace_critical_arclength_binary(model, T0, rhovec0, filename, opt);
        CHECK(j.size() > 3);
        auto tic1 = std::chrono::steady_clock::now();
    }
    
    for (auto ifluid = 0; ifluid < 2; ++ifluid) {
        double T0 = model.redfunc.Tc[ifluid];
        Eigen::ArrayXd rhovec0(2); rhovec0 = 0.0; rhovec0[ifluid] = 1.0 / model.redfunc.vc[ifluid];

        auto tic0 = std::chrono::steady_clock::now();
        std::string filename = "";
        using ct = CriticalTracing<decltype(model), double, Eigen::ArrayXd>;
        TCABOptions opt; opt.max_dt = 10000; opt.init_dt = 10; opt.abs_err = 1e-8; opt.rel_err = 1e-6; opt.small_T_count = 100;
        auto j = ct::trace_critical_arclength_binary(model, T0, rhovec0, filename, opt);
        CHECK(j.size() > 3);
        auto tic1 = std::chrono::steady_clock::now();
    }
}

TEST_CASE("Check that all pure fluid models can be instantiated", "[multifluid],[all]"){
    std::string root = "../mycp";
    SECTION("With absolute paths to json file") {
        int counter = 0;
        for (auto path : get_files_in_folder(root + "/dev/fluids", ".json")) {
            if (path.filename().stem() == "Methanol") { continue; }
            CAPTURE(path.string());
            auto abspath = std::filesystem::absolute(path).string();
            auto model = build_multifluid_model({ abspath }, root, root + "/dev/mixtures/mixture_binary_pairs.json");
            std::valarray<double> z(0.0, 1);
            model.alphar(300, 1.0, z);
            counter += 1;
        }
        CHECK(counter > 100);
    }
    SECTION("With filename stems") {
        for (auto path : get_files_in_folder(root + "/dev/fluids", ".json")) {
            auto stem = path.filename().stem().string(); // filename without the .json
            if (stem == "Methanol") { continue; }
            auto model = build_multifluid_model({ stem }, root, root + "/dev/mixtures/mixture_binary_pairs.json");
            std::valarray<double> z(0.0, 1);
            model.alphar(300, 1.0, z);
        }
    }    
}

TEST_CASE("Check that mixtures can also do absolute paths", "[multifluid],[abspath]") {
    std::string root = "../mycp";
    SECTION("With absolute paths to json file") {
        std::vector<std::filesystem::path> paths = { root + "/dev/fluids/Methane.json", root + "/dev/fluids/Ethane.json" };
        std::vector<std::string> abspaths;
        for (auto p : paths) {
            abspaths.emplace_back(std::filesystem::absolute(p).string());
        }
        auto model = build_multifluid_model(abspaths, root, root + "/dev/mixtures/mixture_binary_pairs.json");
        auto model2 = build_multifluid_model(abspaths, root); // default path for BIP
    }
}

TEST_CASE("Check mixing absolute and relative paths and fluid names", "[multifluid],[abspath]") {
    std::string root = "../mycp";
    SECTION("With correct name of fluid") {
        std::vector<std::string> paths = { std::filesystem::absolute(root + "/dev/fluids/Methane.json").string(), "Ethane" };
        auto model = build_multifluid_model(paths, root, root + "/dev/mixtures/mixture_binary_pairs.json");
    }
    SECTION("Needing a reverse lookup for one fluid") {
        std::vector<std::string> paths = { std::filesystem::absolute(root + "/dev/fluids/Methane.json").string(), "PROPANE" };
        auto model = build_multifluid_model(paths, root, root + "/dev/mixtures/mixture_binary_pairs.json");
    }
}

TEST_CASE("Check that all binary pairs specified in the binary pair file can be instantiated", "[multifluid],[binaries]") {
    std::string root = "../mycp";
    REQUIRE_NOTHROW(build_alias_map(root));
    auto amap = build_alias_map(root);
    for (auto el : load_a_JSON_file(root + "/dev/mixtures/mixture_binary_pairs.json")) {
        auto is_unsupported = [](const auto& s) {
            return (s == "METHANOL" || s == "R1216" || s == "C14" || s == "IOCTANE" || s == "C4F10" || s == "C5F12" || s == "C1CC6" || s == "C3CC6" || s == "CHLORINE" || s == "RE347MCC");
        };
        if (is_unsupported(el["Name1"]) || is_unsupported(el["Name2"])) {
            continue;
        }
        CAPTURE(el["Name1"]);
        CAPTURE(el["Name2"]);
        CHECK_NOTHROW(build_multifluid_model({ amap[el["Name1"]], amap[el["Name2"]] }, root)); // default path for BIP
    }
}

TEST_CASE("Check that all pure fluid models can be evaluated at zero density", "[multifluid],[all],[virial]") {
    std::string root = "../mycp";
    SECTION("With filename stems") {
        for (auto path : get_files_in_folder(root + "/dev/fluids", ".json")) {
            auto stem = path.filename().stem().string(); // filename without the .json
            if (stem == "Methanol") { continue; }
            auto model = build_multifluid_model({ stem }, root);
            std::valarray<double> z(1.0, 1); 
            using tdx = TDXDerivatives<decltype(model), double, decltype(z) >;
            auto ders = tdx::template get_Ar0n<4>(model, model.redfunc.Tc[0], 0.0, z);
            CAPTURE(stem);
            CHECK(std::isfinite(ders[1]));

            using vd = VirialDerivatives<decltype(model),double, decltype(z)>;
            auto Bn = vd::get_Bnvir<4>(model, model.redfunc.Tc[0], z);

            CAPTURE(stem);
            CHECK(std::isfinite(Bn[2]));
        }
    }
}