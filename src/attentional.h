#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "expr-xtra.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

#define RNN_H0_IS_ZERO

namespace cnn {

template <class Builder>
struct AttentionalModel {
    explicit AttentionalModel(Model* model, 
            unsigned vocab_size_src, unsigned vocab_size_tgt, unsigned layers,
            unsigned hidden_dim, unsigned align_dim, bool rnn_src_embeddings,
	    bool giza_positional, bool giza_markov, bool giza_fertility, 
	    bool doc_context, bool global_fertility);

    ~AttentionalModel();

    Expression BuildGraph(const std::vector<int>& source, const std::vector<int>& target, 
            ComputationGraph& cg, Expression* alignment=0, const std::vector<int>* ctx=0,
            Expression *coverage=0, Expression *fertility=0);

    void display_ascii(const std::vector<int> &source, const std::vector<int>& target, 
            ComputationGraph& cg, const Expression& alignment, Dict &sd, Dict &td);

    void display_tikz(const std::vector<int> &source, const std::vector<int>& target, 
            ComputationGraph& cg, const Expression& alignment, Dict &sd, Dict &td);

    void display_fertility(const std::vector<int> &source, Dict &sd);

    void display_empirical_fertility(const std::vector<int> &source, const std::vector<int> &target, Dict &sd);

    std::vector<int> greedy_decode(const std::vector<int> &source, ComputationGraph& cg, 
            Dict &tdict, const std::vector<int>* ctx=0);

    std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, 
            int beam_width, Dict &tdict, const std::vector<int>* ctx=0);

    std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg, 
            Dict &tdict, const std::vector<int>* ctx=0);

    void add_fertility_params(cnn::Model* model, unsigned hidden_dim, bool _rnn_src_embeddings);

    LookupParameter p_cs;
    LookupParameter p_ct;
    Parameter p_R;
    Parameter p_Q;
    Parameter p_P;
    Parameter p_S;
    Parameter p_bias;
    Parameter p_Wa;
    std::vector<Parameter> p_Wh0;
    Parameter p_Ua;
    Parameter p_va;
    Parameter p_Ta;
    Parameter p_Wfhid;
    Parameter p_Wfmu;
    Parameter p_Wfvar;
    Parameter p_bfhid;
    Parameter p_bfmu;
    Parameter p_bfvar;
    Builder builder;
    Builder builder_src_fwd;
    Builder builder_src_bwd;
    bool rnn_src_embeddings;
    bool giza_positional;
    bool giza_markov;
    bool giza_fertility;
    bool doc_context;
    bool global_fertility;
    unsigned vocab_size_tgt;

    // statefull functions for incrementally creating computation graph, one
    // target word at a time
    void start_new_instance(const std::vector<int> &src, ComputationGraph &cg, const std::vector<int> *ctx=0);
    Expression add_input(int tgt_tok, int t, ComputationGraph &cg, RNNPointer *prev_state=0);
    std::vector<float> *auxiliary_vector(); // memory management

    // state variables used in the above two methods
    Expression src;
    Expression i_R;
    Expression i_Q;
    Expression i_P;
    Expression i_S;
    Expression i_bias;
    Expression i_Wa;
    Expression i_Ua;
    Expression i_va;
    Expression i_uax;
    Expression i_Ta;
    Expression i_src_idx;
    Expression i_src_len;
    Expression i_tt_ctx;
    std::vector<Expression> aligns;
    std::vector<std::vector<float>*> aux_vecs; // special storage for constant vectors
    unsigned num_aux_vecs;
    unsigned slen;
    bool has_document_context;
};

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

template <class Builder>
AttentionalModel<Builder>::AttentionalModel(cnn::Model* model,
    unsigned vocab_size_src, unsigned _vocab_size_tgt, unsigned layers, unsigned hidden_dim, 
    unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional, 
    bool _giza_markov, bool _giza_fertility, bool _doc_context,
    bool _global_fertility)
: builder(layers, (_rnn_src_embeddings) ? 3*hidden_dim : 2*hidden_dim, hidden_dim, model),
  builder_src_fwd(1, hidden_dim, hidden_dim, model),
  builder_src_bwd(1, hidden_dim, hidden_dim, model),
  rnn_src_embeddings(_rnn_src_embeddings), 
  giza_positional(_giza_positional), giza_markov(_giza_markov), giza_fertility(_giza_fertility),
  doc_context(_doc_context),
  global_fertility(_global_fertility),
  vocab_size_tgt(_vocab_size_tgt),
  num_aux_vecs(0)
{
    //std::cerr << "Attentionalmodel(" << vocab_size_src  << " " <<  _vocab_size_tgt  << " " <<  layers  << " " <<  hidden_dim << " " <<  align_dim  << " " <<  _rnn_src_embeddings  << " " <<  _giza_extentions  << " " <<  _doc_context << ")\n";
    
    p_cs = model->add_lookup_parameters(vocab_size_src, {hidden_dim}); 
    p_ct = model->add_lookup_parameters(vocab_size_tgt, {hidden_dim}); 
    p_R = model->add_parameters({vocab_size_tgt, hidden_dim});
    p_P = model->add_parameters({hidden_dim, hidden_dim});
    p_bias = model->add_parameters({vocab_size_tgt});
    p_Wa = model->add_parameters({align_dim, layers*hidden_dim});
    if (rnn_src_embeddings) {
        p_Ua = model->add_parameters({align_dim, 2*hidden_dim});
	p_Q = model->add_parameters({hidden_dim, 2*hidden_dim});
    } else {
        p_Ua = model->add_parameters({align_dim, hidden_dim});
	p_Q = model->add_parameters({hidden_dim, hidden_dim});
    }
    if (giza_positional || giza_markov || giza_fertility) {
        int num_giza = 0;
        if (giza_positional) num_giza += 3;
        if (giza_markov) num_giza += 3;
        if (giza_fertility) num_giza += 3;
        p_Ta = model->add_parameters({align_dim, (unsigned int)num_giza});
    }
    p_va = model->add_parameters({align_dim});

    if (doc_context) {
        if (rnn_src_embeddings) {
            p_S = model->add_parameters({hidden_dim, 2*hidden_dim});
        } else {
            p_S = model->add_parameters({hidden_dim, hidden_dim});
        }
    }

    if (global_fertility) {
        if (rnn_src_embeddings) {
            p_Wfhid = model->add_parameters({hidden_dim, 2*hidden_dim});
        } else {
            p_Wfhid = model->add_parameters({hidden_dim, hidden_dim});
        }
        p_bfhid = model->add_parameters({hidden_dim});
        p_Wfmu = model->add_parameters({hidden_dim});
        p_bfmu = model->add_parameters({1});
        p_Wfvar = model->add_parameters({hidden_dim});
        p_bfvar = model->add_parameters({1});
    }

    int hidden_layers = builder.num_h0_components();
    for (int l = 0; l < hidden_layers; ++l) {
	if (rnn_src_embeddings)
	    p_Wh0.push_back(model->add_parameters({hidden_dim, 2*hidden_dim}));
	else
	    p_Wh0.push_back(model->add_parameters({hidden_dim, hidden_dim}));
    }
}

template <class Builder>
void AttentionalModel<Builder>::add_fertility_params(cnn::Model* model, unsigned hidden_dim, bool _rnn_src_embeddings)
{
    if (_rnn_src_embeddings) {
         p_Wfhid = model->add_parameters({hidden_dim, 2*hidden_dim});
     } else {
         p_Wfhid = model->add_parameters({hidden_dim, hidden_dim});
     }
     p_bfhid = model->add_parameters({hidden_dim});
     p_Wfmu = model->add_parameters({hidden_dim});
     p_bfmu = model->add_parameters({1});
     p_Wfvar = model->add_parameters({hidden_dim});
     p_bfvar = model->add_parameters({1});

     global_fertility = true;
}

template <class Builder>
AttentionalModel<Builder>::~AttentionalModel()
{
    for (auto v: aux_vecs)
        delete v;
}

template <class Builder>
std::vector<float>* AttentionalModel<Builder>::auxiliary_vector()
{
    while (num_aux_vecs >= aux_vecs.size())
        aux_vecs.push_back(new std::vector<float>());
    // NB, we return the last auxiliary vector, AND increment counter
    return aux_vecs[num_aux_vecs++]; 
}

template <class Builder>
void AttentionalModel<Builder>::start_new_instance(const std::vector<int> &source, ComputationGraph &cg, const std::vector<int> *ctx)
{
    //slen = source.size() - 1; 
    slen = source.size(); 
    std::vector<Expression> source_embeddings;
    if (!rnn_src_embeddings) {
	for (unsigned s = 0; s < slen; ++s) 
	    source_embeddings.push_back(lookup(cg, p_cs, source[s]));
    } else {
	// run a RNN backward and forward over the source sentence
	// and stack the top-level hidden states from each model as 
	// the representation at each position
	std::vector<Expression> src_fwd(slen);
	builder_src_fwd.new_graph(cg);
	builder_src_fwd.start_new_sequence();
	for (unsigned i = 0; i < slen; ++i) 
	    src_fwd[i] = builder_src_fwd.add_input(lookup(cg, p_cs, source[i]));

	std::vector<Expression> src_bwd(slen);
	builder_src_bwd.new_graph(cg);
	builder_src_bwd.start_new_sequence();
	for (int i = slen-1; i >= 0; --i) {
	    // offset by one position to the right, to catch </s> and generally
	    // not duplicate the w_t already captured in src_fwd[t]
	    src_bwd[i] = builder_src_bwd.add_input(lookup(cg, p_cs, source[i]));
	}

	for (unsigned i = 0; i < slen; ++i) 
	    source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
    }
    src = concatenate_cols(source_embeddings); 

    // now for the target sentence
    i_R = parameter(cg, p_R); // hidden -> word rep parameter
    i_Q = parameter(cg, p_Q);
    i_P = parameter(cg, p_P);
    i_bias = parameter(cg, p_bias);  // word bias
    i_Wa = parameter(cg, p_Wa); 
    i_Ua = parameter(cg, p_Ua);
    i_va = parameter(cg, p_va);
    i_uax = i_Ua * src; 

    // reset aux_vecs counter, allowing the memory to be reused
    num_aux_vecs = 0;

    if (giza_fertility || giza_markov || giza_positional) {
	i_Ta = parameter(cg, p_Ta);   
        if (giza_positional) {
            i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
            i_src_len = repeat(cg, slen, log(1.0 + slen), auxiliary_vector());
        }
    }

    aligns.clear();
    aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

    // initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
    std::vector<Expression> h0;
    Expression i_src = average(source_embeddings); // try max instead?
    int hidden_layers = builder.num_h0_components();
    for (int l = 0; l < hidden_layers; ++l) {
	Expression i_Wh0 = parameter(cg, p_Wh0[l]);
	h0.push_back(tanh(i_Wh0 * i_src));
    }
    builder.new_graph(cg); 
    builder.start_new_sequence(h0);
#else
    builder.new_graph(cg); 
    builder.start_new_sequence();
#endif

    // document context; n.b. use "0" context for the first sentence
    if (doc_context && ctx != 0) { 
        const std::vector<int> &context = *ctx;

        std::vector<Expression> ctx_embed;
        if (!rnn_src_embeddings) {
            for (unsigned s = 1; s+1 < context.size(); ++s) 
                ctx_embed.push_back(lookup(cg, p_cs, context[s]));
        } else {
            ctx_embed.resize(context.size()-1);
            builder_src_fwd.start_new_sequence();
            for (unsigned i = 0; i+1 < context.size(); ++i) 
                ctx_embed[i] = builder_src_fwd.add_input(lookup(cg, p_cs, context[i]));
        }
        Expression avg_context = average(source_embeddings); 
        i_S = parameter(cg, p_S);
        i_tt_ctx = i_S * avg_context;
        has_document_context = true;
    } else {
        has_document_context = false;
    }
}

template <class Builder>
Expression AttentionalModel<Builder>::add_input(int trg_tok, int t, ComputationGraph &cg, RNNPointer *prev_state)
{
    // alignment input 
    Expression i_wah_rep;
    if (t > 0) {
	//auto i_h_tm1 = builder.final_h().back();
	auto i_h_tm1 = concatenate(builder.final_h());
	Expression i_wah = i_Wa * i_h_tm1;
	// want numpy style broadcasting, but have to do this manually
	i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
    }

    Expression i_e_t;
    if (giza_markov || giza_fertility || giza_positional) {
	std::vector<Expression> alignment_context;
	if (giza_markov || giza_fertility) {
            if (t > 0) {
                if (giza_fertility) {
                    auto i_aprev = concatenate_cols(aligns);
                    auto i_asum = sum_cols(i_aprev);
                    auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());
                    alignment_context.push_back(i_asum_pm);
                }
                if (giza_markov) {
                    auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
                    alignment_context.push_back(i_alast_pm);
                }
            } else {
                // just 6 repeats of the 0 vector
                auto zeros = repeat(cg, slen, 0, auxiliary_vector());
                if (giza_fertility) {
                    alignment_context.push_back(zeros); 
                    alignment_context.push_back(zeros);
                    alignment_context.push_back(zeros);
                }
                if (giza_markov) {
                    alignment_context.push_back(zeros);
                    alignment_context.push_back(zeros);
                    alignment_context.push_back(zeros);
                }
            }
        }
        if (giza_positional) {
            alignment_context.push_back(i_src_idx);
            alignment_context.push_back(i_src_len);
            auto i_tgt_idx = repeat(cg, slen, log(1.0 + t), auxiliary_vector());
            alignment_context.push_back(i_tgt_idx);
        }
	auto i_context = concatenate_cols(alignment_context);

	auto i_e_t_input = i_uax + i_Ta * transpose(i_context); 
	if (t > 0) i_e_t_input = i_e_t_input + i_wah_rep;
	i_e_t = transpose(tanh(i_e_t_input)) * i_va;
    } else {
	if (t > 0) 
	    i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
	else
	    i_e_t = transpose(tanh(i_uax)) * i_va;
    }
    Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
    aligns.push_back(i_alpha_t);
    Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?
    // word input
    Expression i_x_t = lookup(cg, p_ct, trg_tok);
    Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t})); 
    // y_t = RNN([x_t, a_t])
    Expression i_y_t;
    if (prev_state)
       i_y_t = builder.add_input(*prev_state, input);
    else
       i_y_t = builder.add_input(input);

    if (doc_context && has_document_context)
        i_y_t = i_y_t + i_tt_ctx;
#ifndef VANILLA_TARGET_LSTM
    // Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
    Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
    Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t}); 
#else
    Expression i_r_t = affine_transform({i_bias, i_R, i_y_t}); 
#endif

    return i_r_t;
}

template <class Builder>
Expression AttentionalModel<Builder>::BuildGraph(const std::vector<int> &source,
        const std::vector<int>& target, ComputationGraph& cg, Expression *alignment,
        const std::vector<int>* ctx, Expression *coverage, Expression *fertility) 
{
    //std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
    start_new_instance(source, cg, ctx);

    std::vector<Expression> errs;
    const unsigned tlen = target.size() - 1; 
    for (unsigned t = 0; t < tlen; ++t) {
        Expression i_r_t = add_input(target[t], t, cg);
        Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
        errs.push_back(i_err);
    }
    // save the alignment for later
    if (alignment != 0) {
	// pop off the last alignment column
        *alignment = concatenate_cols(aligns);
    }

    // AM paper (vision one) has a penalty over alignment rows deviating from 1
    if (coverage != nullptr || fertility != nullptr) {
        Expression i_aligns = (alignment != 0) ? *alignment : concatenate_cols(aligns);
        Expression i_totals = sum_cols(i_aligns);
        // only care about the non-null entries
        Expression i_total_trim = pickrange(i_totals, 1, slen-1);

        if (coverage != nullptr) {
            Expression i_ones = repeat(cg, slen-2, 1.0f, auxiliary_vector());
            Expression i_penalty = squared_distance(i_total_trim, i_ones);
            *coverage = i_penalty;
        } 

        if (fertility != nullptr) {
            assert(global_fertility);

            Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
            Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
            Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
            Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));  
            Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
            Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));

            Expression mu_trim = pickrange(mu, 1, slen-1);
            Expression var_trim = pickrange(var, 1, slen-1);

#if 0
            /* log-Normal distribution */
            Expression log_fert = log(i_total_trim);
            Expression delta = log_fert - mu_trim;
            Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
            Expression partition = -log_fert - 0.5 * log(2.0f * var_trim * 3.14159265359);
            *fertility = -sum_cols(transpose(partition + exponent));
#else
            /* Normal distribution */
            Expression delta = i_total_trim - mu_trim;
            Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
            Expression partition = -0.5 * log(2.0f * var_trim * 3.14159265359);
            *fertility = -sum_cols(transpose(partition + exponent));
            // note that as this is the value of the normal density, the errors
            // are not strictly positive
#endif

            //LOLCAT(transpose(i_total_trim));
            //LOLCAT(transpose(mu_trim));
            //LOLCAT(transpose(var_trim));
            //LOLCAT(transpose(partition + exponent));
            //LOLCAT(exp(transpose(partition + exponent)));
        }
    }

    Expression i_nerr = sum(errs);
    return i_nerr;
}

template <class Builder>
void 
AttentionalModel<Builder>::display_ascii(const std::vector<int> &source, const std::vector<int>& target, 
                          ComputationGraph &cg, const Expression &alignment, Dict &sd, Dict &td)
{
    using namespace std;

    // display the alignment
    //float I = target.size() - 1;
    //float J = source.size() - 1;
    float I = target.size();
    float J = source.size();
    //vector<string> symbols{"\u2588","\u2589","\u258A","\u258B","\u258C","\u258D","\u258E","\u258F"};
    vector<string> symbols{".","o","*","O","@"};
    int num_symbols = symbols.size();
    vector<float> thresholds;
    thresholds.push_back(0.8/I);
    float lgap = (0 - log(thresholds.back())) / (num_symbols - 1);
    for (auto rit = symbols.begin(); rit != symbols.end(); ++rit) {
        float thr = exp(log(thresholds.back()) + lgap);
        thresholds.push_back(thr);
    }
    // FIXME: thresholds > 1, what's going on?
    //cout << thresholds.back() << endl;

    const Tensor &a = cg.get_value(alignment.i);
    //cout << "I = " << I << " J = " << J << endl;

    cout.setf(ios_base::adjustfield, ios_base::left);
    cout << setw(12) << "source" << "  ";
    cout.setf(ios_base::adjustfield, ios_base::right);
    for (int j = 0; j < J; ++j) 
        cout << setw(2) << j << ' ';
    cout << endl;

    for (int i = 0; i < I; ++i) {
        cout.setf(ios_base::adjustfield, ios_base::left);
        //cout << setw(12) << td.convert(target[i+1]) << "  ";
        cout << setw(12) << td.convert(target[i]) << "  ";
        cout.setf(ios_base::adjustfield, ios_base::right);
        float max_v = 0;
        int max_j = -1;
        for (int j = 0; j < J; ++j) {
            float v = TensorTools::AccessElement(a, Dim({(unsigned int)j, (unsigned int)i}));
            string symbol;
            for (int s = 0; s <= num_symbols; ++s) {
                if (s == 0) 
                    symbol = ' ';
                else
                    symbol = symbols[s-1];
                if (s != num_symbols && v < thresholds[s])
                    break;
            }
            cout << setw(2) << symbol << ' ';
            if (v >= max_v) {
                max_v = v;
                max_j = j;
            }
        }
        cout << setw(20) << "max Pr=" << setprecision(3) << setw(5) << max_v << " @ " << max_j << endl;
    }
    cout << resetiosflags(ios_base::adjustfield);
    for (int j = 0; j < J; ++j) 
        cout << j << ":" << sd.convert(source[j]) << ' ';
    cout << endl;
}

template <class Builder>
void 
AttentionalModel<Builder>::display_tikz(const std::vector<int> &source, const std::vector<int>& target, 
                          ComputationGraph &cg, const Expression &alignment, Dict &sd, Dict &td)
{
    using namespace std;

    // display the alignment
    float I = target.size();
    float J = source.size();

    const Tensor &a = cg.get_value(alignment.i);
    cout << a.d[0] << " x " << a.d[1] << endl;

    cout << "\\begin{tikzpicture}[scale=0.5]\n";
    for (int j = 0; j < J; ++j) 
        cout << "\\node[anchor=west,rotate=90] at (" << j+0.5 << ", " << I+0.2 << ") { " << sd.convert(source[j]) << " };\n";
    for (int i = 0; i < I; ++i) 
        cout << "\\node[anchor=west] at (" << J+0.2 << ", " << I-i-0.5 << ") { " << td.convert(target[i]) << " };\n";

    float eps = 0.01;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            float v = TensorTools::AccessElement(a, Dim({(unsigned int)j, (unsigned int)i}));
            //int val = int(pow(v, 0.5) * 100);
            int val = int(v * 100);
            cout << "\\fill[blue!" << val << "!black] (" << j+eps << ", " << I-i-1+eps << ") rectangle (" << j+1-eps << "," << I-i-eps << ");\n";
        }
    }
    cout << "\\draw[step=1cm,color=gray] (0,0) grid (" << J << ", " << I << ");\n";
    cout << "\\end{tikzpicture}\n";
}


template <class Builder>
std::vector<int>
AttentionalModel<Builder>::greedy_decode(const std::vector<int> &source, ComputationGraph& cg, 
        cnn::Dict &tdict, const std::vector<int>* ctx)
{
    const int sos_sym = tdict.convert("<s>");
    const int eos_sym = tdict.convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym); 

    //std::cerr << tdict.convert(target.back());
    int t = 0;
    start_new_instance(source, cg, ctx);
    while (target.back() != eos_sym) 
    {
        Expression i_scores = add_input(target.back(), t, cg);
        Expression ydist = softmax(i_scores); // compiler warning, but see below

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
        }

        // break potential infinite loop
        if (t > 2*source.size()) {
            w = eos_sym;
            pr_w = dist[w];
        }

        //std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
        t += 1;
        target.push_back(w);
    }
    //std::cerr << std::endl;

    return target;
}

struct Hypothesis {
    Hypothesis() {};
    Hypothesis(RNNPointer state, int tgt, float cst, std::vector<Expression> &al)
        : builder_state(state), target({tgt}), cost(cst), aligns(al) {}
    Hypothesis(RNNPointer state, int tgt, float cst, Hypothesis &last, std::vector<Expression> &al)
        : builder_state(state), target(last.target), cost(cst), aligns(al) {
        target.push_back(tgt);
    }
    RNNPointer builder_state;
    std::vector<int> target;
    float cost;
    std::vector<Expression> aligns;
};

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, 
        cnn::Dict &tdict, const std::vector<int>* ctx)
{
    const int sos_sym = tdict.convert("<s>");
    const int eos_sym = tdict.convert("</s>");

    start_new_instance(source, cg, ctx);

    std::vector<Hypothesis> chart;
    chart.push_back(Hypothesis(builder.state(), sos_sym, 0.0f, aligns));

    std::vector<unsigned> vocab(boost::copy_range<std::vector<unsigned>>(boost::irange(0u, vocab_size_tgt)));
    std::vector<Hypothesis> completed;

    for (int steps = 0; completed.size() < beam_width && steps < 2*source.size(); ++steps) {
        std::vector<Hypothesis> new_chart;

        for (auto &hprev: chart) {
            //std::cerr << "hypo t[-1]=" << tdict.convert(hprev.target.back()) << " cost " << hprev.cost << std::endl;
            if (giza_markov || giza_fertility) 
                aligns = hprev.aligns;
            Expression i_scores = add_input(hprev.target.back(), hprev.target.size()-1, cg, &hprev.builder_state);
            Expression ydist = softmax(i_scores); // compiler warning, but see below

            // find the top k best next words
            unsigned w = 0;
            auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
            std::partial_sort(vocab.begin(), vocab.begin()+beam_width, vocab.end(), 
                    [&dist](unsigned v1, unsigned v2) { return dist[v1] > dist[v2]; });

            // add to chart
            for (auto vi = vocab.begin(); vi < vocab.begin() + beam_width; ++vi) {
                //std::cerr << "\t++word " << tdict.convert(*vi) << " prob " << dist[*vi] << std::endl;
                //if (new_chart.size() < beam_width) {
                    Hypothesis hnew(builder.state(), *vi, hprev.cost-log(dist[*vi]), hprev, aligns);
                    if (*vi == eos_sym)
                        completed.push_back(hnew);
                    else
                        new_chart.push_back(hnew);
                //} 
            }
        }

        if (new_chart.size() > beam_width) {
            // sort new_chart by score, to get kbest candidates
            std::partial_sort(new_chart.begin(), new_chart.begin()+beam_width, new_chart.end(),
                    [](Hypothesis &h1, Hypothesis &h2) { return h1.cost < h2.cost; });
            new_chart.resize(beam_width);
        }
        chart.swap(new_chart);
    }

    // sort completed by score, adjusting for length -- not very effective, too short!
    auto best = std::min_element(completed.begin(), completed.end(),
            [](Hypothesis &h1, Hypothesis &h2) { return h1.cost/h1.target.size() < h2.cost/h2.target.size(); });
    assert(best != completed.end());

    return best->target;
}

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::sample(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict &tdict,
        const std::vector<int> *ctx)
{
    const int sos_sym = tdict.convert("<s>");
    const int eos_sym = tdict.convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym); 

    std::cerr << tdict.convert(target.back());
    int t = 0;
    start_new_instance(source, cg, ctx);
    while (target.back() != eos_sym) 
    {
        Expression i_scores = add_input(target.back(), t, cg);
        Expression ydist = softmax(i_scores);

	// in rnnlm.cc there's a loop around this block -- why? can incremental_forward fail?
        auto dist = as_vector(cg.incremental_forward());
	double p = rand01();
        unsigned w = 0;
        for (; w < dist.size(); ++w) {
	    p -= dist[w];
	    if (p < 0) break;
        }
	// this shouldn't happen
	if (w == dist.size()) w = eos_sym;

        std::cerr << " " << tdict.convert(w) << " [p=" << dist[w] << "]";
        t += 1;
        target.push_back(w);
    }
    std::cerr << std::endl;

    return target;
}

template <class Builder>
void
AttentionalModel<Builder>::display_fertility(const std::vector<int> &source, Dict &sd)
{
    ComputationGraph cg;
    start_new_instance(source, cg);
    assert(global_fertility);

    Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
    Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
    Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
    Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));  
    Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
    auto mu_vec = as_vector(cg.incremental_forward()); // evaluates last expression
    Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));
    auto var_vec = as_vector(cg.incremental_forward()); // evaluates last expression

    for (int j = 1; j < slen-1; ++j) 
        std::cout << sd.convert(source[j]) << '\t' << mu_vec[j] << '\t' << var_vec[j] << '\n';
}

template <class Builder>
void
AttentionalModel<Builder>::display_empirical_fertility(const std::vector<int> &source, const std::vector<int> &target, Dict &sd)
{
    ComputationGraph cg;
    Expression alignment;
    BuildGraph(source, target, cg, &alignment);

    Expression totals = sum_cols(alignment);
    auto totals_vec = as_vector(cg.incremental_forward()); // evaluates last expression

    for (int j = 0; j < slen; ++j) 
        std::cout << sd.convert(source[j]) << '\t' << totals_vec[j] << '\n';
}

#undef WTF
#undef KTHXBYE
#undef LOLCAT

}; // namespace cnn
