#pragma once

#include "nlohmann/json.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/math/pow_templates.hpp"

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include <tuple>

namespace teqp::algorithms::pure_param_optimization {

struct SatRhoLPoint{
    double T, rhoL_exp, rhoL_guess, rhoV_guess, weight=1.0;
    auto check_fields() const{}
    template<typename Model>
    auto calculate_contribution(const Model& model) const{
        auto rhoLrhoV = model->pure_VLE_T(T, rhoL_guess, rhoV_guess, 10);
        return std::abs(rhoLrhoV[0]-rhoL_exp)*weight;
    }
};

#define stdstringify(s) std::string(#s)

#define SatRhoLPPoint_optionalfields X(T) X(p_exp) X(rhoL_exp) X(rhoL_guess) X(rhoV_guess)
struct SatRhoLPPoint{
    #define X(field) std::optional<double> field;
    SatRhoLPPoint_optionalfields
    #undef X
    double weight_rho=1.0, weight_p=1.0, R=8.31446261815324;
    Eigen::ArrayXd z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    auto check_fields() const{
        #define X(field) if (!field){ throw teqp::InvalidArgument("A field [" + stdstringify(field) + "] has not been initialized"); }
        SatRhoLPPoint_optionalfields
        #undef X
    }
    
    template<typename Model>
    auto calculate_contribution(const Model& model) const{
        auto rhoLrhoV = model->pure_VLE_T(T.value(), rhoL_guess.value(), rhoV_guess.value(), 10);
        auto rhoL = rhoLrhoV[0];
        auto p = rhoL*R*T.value()*(1+model->get_Ar01(T.value(), rhoL, z));
//        std::cout << p << "," << p_exp << "," << (p_exp-p)/p_exp << std::endl;
        return std::abs(rhoL-rhoL_exp.value())/rhoL_exp.value()*weight_rho + std::abs(p-p_exp.value())/p_exp.value()*weight_p;
    }
};

#define SatRhoLPWPoint_optionalfields X(T) X(p_exp) X(rhoL_exp) X(w_exp) X(rhoL_guess) X(rhoV_guess) X(Ao20) X(M) X(R)
struct SatRhoLPWPoint{
    
    #define X(field) std::optional<double> field;
    SatRhoLPWPoint_optionalfields
    #undef X
    double weight_rho=1.0, weight_p=1.0, weight_w=1.0;
    Eigen::ArrayXd z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    auto check_fields() const{
        #define X(field) if (!field){ throw teqp::InvalidArgument("A field [" + stdstringify(field) + "] has not been initialized"); }
        SatRhoLPWPoint_optionalfields
        #undef X
    }
    
    template<typename Model>
    auto calculate_contribution(const Model& model) const{
        
        auto rhoLrhoV = model->pure_VLE_T(T.value(), rhoL_guess.value(), rhoV_guess.value(), 10);
        
        // First part, density
        auto rhoL = rhoLrhoV[0];
        
        auto Ar0n = model->get_Ar02n(T.value(), rhoL, z);
        double Ar01 = Ar0n[1], Ar02 = Ar0n[2];
        auto Ar11 = model->get_Ar11(T.value(), rhoL, z);
        auto Ar20 = model->get_Ar20(T.value(), rhoL, z);
        
        // Second part, presure
        auto p = rhoL*R.value()*T.value()*(1+Ar01);

        // Third part, speed of sound
        //
        // M*w^2/(R*T) where w is the speed of sound
        // from the definition w = sqrt(dp/drho|s)
        double Mw2RT = 1 + 2*Ar01 + Ar02 - POW2(1.0 + Ar01 - Ar11)/(Ao20.value() + Ar20);
        double w = sqrt(Mw2RT*R.value()*T.value()/M.value());
        
//        std::cout << p << "," << p_exp << "," << (p_exp-p)/p_exp << std::endl;
        double cost_rhoL = std::abs(rhoL-rhoL_exp.value())/rhoL_exp.value()*weight_rho;
        double cost_p = std::abs(p-p_exp.value())/p_exp.value()*weight_p;
        double cost_w = std::abs(w-w_exp.value())/w_exp.value()*weight_w;
        return cost_rhoL + cost_p + cost_w;
    }
};
using PureOptimizationContribution = std::variant<SatRhoLPoint, SatRhoLPPoint, SatRhoLPWPoint>;

class PureParameterOptimizer{
private:
    auto make_pointers(const std::vector<std::variant<std::string, std::vector<std::string>>>& pointerstrs){
        std::vector<std::vector<nlohmann::json::json_pointer>> pointers_;
        for (auto & pointer: pointerstrs){
            if (std::holds_alternative<std::string>(pointer)){
                const std::string& s= std::get<std::string>(pointer);
                pointers_.emplace_back(1, nlohmann::json::json_pointer(s));
            }
            else{
                std::vector<nlohmann::json::json_pointer> buffer;
                for (const auto& s : std::get<std::vector<std::string>>(pointer)){
                    buffer.emplace_back(s);
                }
                pointers_.push_back(buffer);
            }
        }
        return pointers_;
    }
public:
    const nlohmann::json jbase;
    std::vector<std::vector<nlohmann::json::json_pointer>> pointers;
    std::vector<PureOptimizationContribution> contributions;
    
    PureParameterOptimizer(const nlohmann::json jbase, const std::vector<std::variant<std::string, std::vector<std::string>>>& pointerstrs) : jbase(jbase), pointers(make_pointers(pointerstrs)){}
    
    void add_one_contribution(const PureOptimizationContribution& cont){
        std::visit([](const auto&c){c.check_fields();}, cont);
        contributions.push_back(cont);
    }
    
    auto prepare_helper_models(const auto& model) const {
        return std::vector<double>{1, 3};
    }
    
    template<typename T>
    nlohmann::json build_JSON(const T& x) const{
        
        if (x.size() != pointers.size()){
            throw teqp::InvalidArgument("sizes don't match");
        };
        nlohmann::json j = jbase;
        for (auto i = 0; i < x.size(); ++i){
            for (const auto& ptr : pointers[i]){
                j.at(ptr) = x[i];
            }
        }
        return j;
    }
    
    template<typename T>
    auto prepare(const T& x) const {
        auto j = build_JSON(x);
        auto model = teqp::cppinterface::make_model(j);
        auto helpers = prepare_helper_models(model);
        return std::make_tuple(std::move(model), helpers);
    }
    
    template<typename T>
    auto cost_function(const T& x) const{
        auto [model, helpers] = prepare(x);
        double cost = 0.0;
        for (const auto& contrib : contributions){
            cost += std::visit([&model](const auto& c){ return c.calculate_contribution(model); }, contrib);
        }
        if (!std::isfinite(cost)){
            return 1e30;
        }
        return cost;
    }
    
    template<typename T>
    auto cost_function_threaded(const T& x, std::size_t Nthreads) {
        boost::asio::thread_pool pool{Nthreads}; // Nthreads in the pool
        const auto [model, helpers] = prepare(x);
        std::valarray<double> buffer(contributions.size());
        std::size_t i = 0;
        for (const auto& contrib : contributions){
            auto& dest = buffer[i];
            auto payload = [&model, &dest, contrib] (){
                dest = std::visit([&model](const auto& c){ return c.calculate_contribution(model); }, contrib);
                if (!std::isfinite(dest)){ dest = 1e30; }
            };
            boost::asio::post(pool, payload);
            i++;
        }
        pool.join();
        double summer = 0.0;
        for (auto i = 0; i < contributions.size(); ++i){
//            std::cout << buffer[i] << std::endl;
            summer += buffer[i];
        }
//        std::cout << summer << std::endl;
        return summer;
    }
    
};

}
