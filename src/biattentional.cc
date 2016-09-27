#include "attentional.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;

unsigned LAYERS = 2;
unsigned HIDDEN_DIM = 512;  // 1024
unsigned ALIGN_DIM = 256;  // 1024
unsigned VOCAB_SIZE_SRC = 0;
unsigned VOCAB_SIZE_TGT = 0;
bool BIDIR = true;
bool GIZA_P = true;
bool GIZA_M = true;
bool GIZA_F = true;
bool DOC = false;
bool FERT = false;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;

#define WTF(expression) \
        cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << endl;

template <class Builder>
struct BidirAttentionalModel {
    AttentionalModel<Builder> s2t_model;
    AttentionalModel<Builder> t2s_model;
    double m_trace_weight;
    Expression s2t_align, t2s_align;
    Expression s2t_xent, t2s_xent, trace_bonus;

    explicit BidirAttentionalModel(Model *model, double trace_weight) 
        : s2t_model(model, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, LAYERS,
		HIDDEN_DIM, ALIGN_DIM, BIDIR, GIZA_P, GIZA_M, GIZA_F, DOC, FERT),
          t2s_model(model, VOCAB_SIZE_TGT, VOCAB_SIZE_SRC, LAYERS, 
	        HIDDEN_DIM, ALIGN_DIM, BIDIR, GIZA_P, GIZA_M, GIZA_F, DOC, FERT),
	  m_trace_weight(trace_weight)
    {
        m_trace_weight = trace_weight;   
    }

    void add_fertility_params(cnn::Model* model)
    {
    	s2t_model.add_fertility_params(model, HIDDEN_DIM, BIDIR);
    	t2s_model.add_fertility_params(model, HIDDEN_DIM, BIDIR);
    }

    // return Expression of total loss
    Expression build_graph(const vector<int> &source, const vector<int>& target, ComputationGraph& cg)
    {
        // FIXME: slightly wasteful, the embeddings of the source and target are done twice each
        s2t_xent = s2t_model.BuildGraph(source, target, cg, &s2t_align);
        t2s_xent = t2s_model.BuildGraph(target, source, cg, &t2s_align);

        trace_bonus = trace_of_product(t2s_align, transpose(s2t_align));
	//cout << "xent: src=" << *cg.get_value(src_xent.i) << " tgt=" << *cg.get_value(tgt_xent.i) << endl;
	//cout << "trace bonus: " << *cg.get_value(trace_bonus.i) << endl;

        return s2t_xent + t2s_xent - m_trace_weight * trace_bonus;
        //return src_xent + tgt_xent;
    }

    void initialise(const std::string &src_file, const std::string &tgt_file, Model &model)
    {
	Model sm, tm;
	AttentionalModel<Builder> smb(&sm, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, 
		LAYERS, HIDDEN_DIM, ALIGN_DIM, BIDIR, GIZA_P, GIZA_M, GIZA_F, DOC, FERT);
	AttentionalModel<Builder> tmb(&tm, VOCAB_SIZE_TGT, VOCAB_SIZE_SRC,
		LAYERS, HIDDEN_DIM, ALIGN_DIM, BIDIR, GIZA_P, GIZA_M, GIZA_F, DOC, FERT);

	//for (const auto &p : sm.lookup_parameters_list())  
            //std::cerr << "\tlookup size: " << p->values[0].d << " number: " << p->values.size() << std::endl;
	//for (const auto &p : sm.parameters_list())  
            //std::cerr << "\tparam size: " << p->values.d << std::endl;

        std::cerr << "... loading " << src_file << " ..." << std::endl;
	{
	    ifstream in(src_file);
	    boost::archive::text_iarchive ia(in);
	    ia >> sm;
	}

        std::cerr << "... loading " << tgt_file << " ..." << std::endl;
	{
	    ifstream in(tgt_file);
	    boost::archive::text_iarchive ia(in);
	    ia >> tm;
	}
        std::cerr << "... merging parameters ..." << std::endl;

	unsigned lid = 0;
	auto &lparams = model.lookup_parameters_list();
	assert(lparams.size() == 2*sm.lookup_parameters_list().size());
	for (const auto &p : sm.lookup_parameters_list())  {
	    for (unsigned i = 0; i < p->values.size(); ++i) 
		memcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(cnn::real) * p->values[i].d.size());
	    lid++;
	}
	for (const auto &p : tm.lookup_parameters_list()) {
	    for (unsigned i = 0; i < p->values.size(); ++i) 
		memcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(cnn::real) * p->values[i].d.size());
	    lid++;
	}
	assert(lid == lparams.size());

	unsigned did = 0;
	auto &dparams = model.parameters_list();
	for (const auto &p : sm.parameters_list()) 
	    memcpy(dparams[did++]->values.v, &p->values.v[0], sizeof(cnn::real) * p->values.d.size());
	for (const auto &p : tm.parameters_list()) 
	    memcpy(dparams[did++]->values.v, &p->values.v[0], sizeof(cnn::real) * p->values.d.size());
	assert(did == dparams.size());
    }
};

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Corpus;

Corpus read_corpus(const string &filename)
{
    ifstream in(filename);
    assert(in);
    Corpus corpus;
    string line;
    int lc = 0, stoks = 0, ttoks = 0;
    while(getline(in, line)) {
        ++lc;
        Sentence source, target;
        read_sentence_pair(line, &source, &sd, &target, &td);
        corpus.push_back(SentencePair(source, target));
        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
                (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
    return corpus;
}

int main(int argc, char** argv) {
    cnn::initialize(argc, argv);

    // command line processing
    variables_map vm; 
    options_description opts("Allowed options");
    opts.add_options()
        ("help", "print help message")
        ("config,c", value<string>(), "config file specifying additional command line options")
        ("train,t", value<string>(), "file containing training sentences, with "
            "each line consisting of source ||| target.")
        ("devel,d", value<string>(), "file containing development sentences.")
        ("rescore,r", value<string>(), "rescore (source, target) sentence pairs")
        ("parameters,p", value<string>(), "save best parameters to this file")
        ("initialise,i", value<vector<string>>(), "load initial parameters from file")
        ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("align,a", value<int>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
        ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
        ("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
        ("giza", "use GIZA++ style features in attentional components (corresponds to all of the 'gz' options)")
        ("gz-position", "use GIZA++ positional index features")
        ("gz-markov", "use GIZA++ markov context features")
        ("gz-fertility", "use GIZA++ fertility type features")
        ("curriculum", "use 'curriculum' style learning, focusing on easy problems in earlier epochs")
        ("fertility,f", "learn Normal model of word fertility values")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && vm.count("rescore") != 1)) {
        cout << opts << "\n";
        return 1;
    }

    kSRC_SOS = sd.convert("<s>");
    kSRC_EOS = sd.convert("</s>");
    kTGT_SOS = td.convert("<s>");
    kTGT_EOS = td.convert("</s>");

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    vector<SentencePair> training, dev, testing;
    string line;
    cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
    training = read_corpus(vm["train"].as<string>());
    sd.freeze(); // no new word types allowed
    td.freeze(); // no new word types allowed
    VOCAB_SIZE_SRC = sd.size();
    VOCAB_SIZE_TGT = td.size();

    if (vm.count("devel")) {
        cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
        dev = read_corpus(vm["devel"].as<string>());
    }

    if (vm.count("rescore")) {
        cerr << "Reading test data from " << vm["rescore"].as<string>() << "...\n";
        testing = read_corpus(vm["rescore"].as<string>());
    }

    LAYERS = vm["layers"].as<int>(); 
    ALIGN_DIM = vm["align"].as<int>(); 
    HIDDEN_DIM = vm["hidden"].as<int>(); 
    BIDIR = vm.count("bidirectional");
    bool giza = vm.count("giza");
    GIZA_P = giza || vm.count("gz-position");
    GIZA_M = giza || vm.count("gz-markov");
    GIZA_F = giza || vm.count("gz-fertility");
    //FERT = vm.count("fertility");

    string fname;
    if (!vm.count("parameters")) {
        ostringstream os;
        os << "bam"
            << '_' << LAYERS
            << '_' << HIDDEN_DIM
            << '_' << ALIGN_DIM
            << "_lstm"
            << "_b" << BIDIR
	    << "_g" << (int)GIZA_P << (int)GIZA_M << (int)GIZA_F
            << "-pid" << getpid() << ".params";
        fname = os.str();
    }
    else {
        fname = vm["parameters"].as<string>();
    }

    if (!vm.count("initialise")) cerr << "Parameters will be written to: " << fname << endl;

    double best = 9e+99;

    Model model;
    SimpleSGDTrainer sgd(&model);
    BidirAttentionalModel<LSTMBuilder> am(&model, 0.1);

    bool add_fer = false;
    if (vm.count("rescore"))
    {
    	am.add_fertility_params(&model);
    	add_fer = true;
    }

    if (vm.count("initialise")) {
        vector<string> init_files = vm["initialise"].as<vector<string>>();
        if (init_files.size() == 1) {
        	cerr << "Parameters will be loaded from: " << init_files[0] << endl;

            ifstream in(init_files[0]);
            boost::archive::text_iarchive ia(in);
            ia >> model;
        } else if (init_files.size() == 2) {
            cerr << "initialising from " << init_files[0] << " and " << init_files[1] << endl;
            am.initialise(init_files[0], init_files[1], model);
        } else {
            assert(false);
        }
    }

    FERT = vm.count("fertility");
    if (FERT && !add_fer) am.add_fertility_params(&model);

        if (false) {
            double dloss = 0, dloss_s2t = 0, dloss_t2s = 0, dloss_trace = 0;
            unsigned dchars_s = 0, dchars_t = 0, dchars_tt = 0;
            unsigned i = 0;
            for (auto& spair : dev) {
                ComputationGraph cg;
                am.build_graph(spair.first, spair.second, cg);
                dloss += as_scalar(cg.incremental_forward());
                dloss_s2t += as_scalar(cg.get_value(am.s2t_xent.i));
                dloss_t2s += as_scalar(cg.get_value(am.t2s_xent.i));
                dloss_trace += as_scalar(cg.get_value(am.trace_bonus.i));
                dchars_s += spair.first.size() - 1;
                dchars_t += spair.second.size() - 1;
                dchars_tt += std::max(spair.first.size(), spair.second.size()) - 1;

                if (true)
                {
                    cout << "\n===== SENTENCE " << i++ << " =====\n";
                    am.s2t_model.display_tikz(spair.first, spair.second, cg, am.s2t_align, sd, td);
                    cout << "\n";
                    am.t2s_model.display_tikz(spair.second, spair.first, cg, am.t2s_align, td, sd);
                    cout << "\n";
                }
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=0]";
            cerr << " E = " << (dloss / (dchars_s + dchars_t)) << " ppl=" << exp(dloss / (dchars_s + dchars_t)) << ' '; // kind of hacky, as trace should be normalised differently
            cerr << " ppl_s = " << exp(dloss_t2s / dchars_s) << ' ';
            cerr << " ppl_t = " << exp(dloss_s2t / dchars_t) << ' ';
            cerr << " trace = " << exp(dloss_trace / dchars_tt) << ' ';
        }

    if (vm.count("rescore")) {
    	double dloss = 0, dloss_s2t = 0, dloss_t2s = 0, dloss_trace = 0;
    	unsigned dchars_s = 0, dchars_t = 0, dchars_tt = 0;
    	for (unsigned i = 0; i < testing.size(); ++i) {
            ComputationGraph cg;
            am.build_graph(testing[i].first, testing[i].second, cg);

            dchars_s += testing[i].first.size() - 1;
            dchars_t += testing[i].second.size() - 1;
            dchars_tt += std::max(testing[i].first.size(), testing[i].second.size()) - 1; // max or min?

            //cg.forward();
            dloss += as_scalar(cg.forward());

            double loss_s2t = as_scalar(cg.get_value(am.s2t_xent.i));
            double loss_t2s = as_scalar(cg.get_value(am.t2s_xent.i));
            double loss_trace = as_scalar(cg.get_value(am.trace_bonus.i));

            dloss_s2t += loss_s2t;
            dloss_t2s += loss_t2s;
            dloss_trace += loss_trace;

            cout << i << " |||";
            for (auto &w: testing[i].first)
                cout << " " << sd.convert(w);
            cout << " |||";
            for (auto &w: testing[i].second)
                cout << " " << td.convert(w);
            cout << " ||| " << (loss_s2t / (testing[i].second.size()-1))
                << " " << (loss_t2s / (testing[i].first.size()-1))
                << " " << (loss_trace / (std::max(testing[i].first.size(), testing[i].second.size())-1)) 
                << std::endl;
        }

    	cerr <<endl << " E = " << (dloss / (dchars_s + dchars_t)) << " ppl=" << exp(dloss / (dchars_s + dchars_t)) << ' '; // kind of hacky, as trace should be normalised differently
    	cerr << " ppl_s = " << exp(dloss_t2s / dchars_s) << ' ';
    	cerr << " ppl_t = " << exp(dloss_s2t / dchars_t) << ' ';
    	cerr << " trace = " << exp(dloss_trace / dchars_tt) << endl;

        return 0;
    }

    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500; // 500
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    while(1) {
        Timer iteration("completed in");
        double loss = 0, loss_s2t = 0, loss_t2s = 0, loss_trace = 0;
        unsigned chars_s = 0, chars_t = 0, chars_tt = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training.size()) {
                si = 0;
                if (first) { first = false; } else { sgd.update_epoch(); }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance
            ComputationGraph cg;
            auto& spair = training[order[si]];
            chars_s += spair.first.size() - 1;
            chars_t += spair.second.size() - 1;
            chars_tt += std::max(spair.first.size(), spair.second.size()) - 1; // max or min?
            ++si;
            am.build_graph(spair.first, spair.second, cg);
            loss += as_scalar(cg.forward());
            loss_s2t += as_scalar(cg.get_value(am.s2t_xent.i));
            loss_t2s += as_scalar(cg.get_value(am.t2s_xent.i));
            loss_trace += as_scalar(cg.get_value(am.trace_bonus.i));
            cg.backward();
            sgd.update(1.0f);
            ++lines;

            //if ((i+1) == report_every_i) 
            if (si == 1) {
		//Expression aligns = concatenate({transpose(am.src_align), am.tgt_align});
		//cerr << cg.get_value(aligns.i) << "\n";
                am.s2t_model.display_ascii(spair.first, spair.second, cg, am.s2t_align, sd, td);
                am.t2s_model.display_ascii(spair.second, spair.first, cg, am.t2s_align, td, sd);
		cerr << "\txent_s2t " << as_scalar(cg.get_value(am.s2t_xent.i))
		     << "\txent_t2s " << as_scalar(cg.get_value(am.t2s_xent.i))
		     << "\ttrace " << as_scalar(cg.get_value(am.trace_bonus.i)) << endl;
            }
        }
        sgd.status();
        cerr << " E = " << (loss / (chars_s + chars_t)) << " ppl=" << exp(loss / (chars_s + chars_t)) << ' '; // kind of hacky, as trace should be normalised differently
        cerr << " ppl_s = " << exp(loss_t2s / chars_s) << ' ';
        cerr << " ppl_t = " << exp(loss_s2t / chars_t) << ' ';
        cerr << " trace = " << exp(loss_trace / chars_tt) << ' ';

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0, dloss_s2t = 0, dloss_t2s = 0, dloss_trace = 0;
            unsigned dchars_s = 0, dchars_t = 0, dchars_tt = 0;
            for (auto& spair : dev) {
                ComputationGraph cg;
                am.build_graph(spair.first, spair.second, cg);
                dloss += as_scalar(cg.incremental_forward());
                dloss_s2t += as_scalar(cg.get_value(am.s2t_xent.i));
                dloss_t2s += as_scalar(cg.get_value(am.t2s_xent.i));
                dloss_trace += as_scalar(cg.get_value(am.trace_bonus.i));
                dchars_s += spair.first.size() - 1;
                dchars_t += spair.second.size() - 1;
                dchars_tt += std::max(spair.first.size(), spair.second.size()) - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "]";
            cerr << " E = " << (dloss / (dchars_s + dchars_t)) << " ppl=" << exp(dloss / (dchars_s + dchars_t)) << ' '; // kind of hacky, as trace should be normalised differently
            cerr << " ppl_s = " << exp(dloss_t2s / dchars_s) << ' ';
            cerr << " ppl_t = " << exp(dloss_s2t / dchars_t) << ' ';
            cerr << " trace = " << exp(dloss_trace / dchars_tt) << ' ';
        }
    }

    return 0;
}

