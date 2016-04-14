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

unsigned LAYERS = 1; // 2
unsigned HIDDEN_DIM = 64;  // 1024
unsigned ALIGN_DIM = 32;   // 128
unsigned SRC_VOCAB_SIZE = 0;
unsigned TGT_VOCAB_SIZE = 0;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
bool verbose;

typedef vector<int> Sentence;
//typedef pair<Sentence, Sentence> SentencePair;
typedef tuple<Sentence, Sentence, int> SentencePair; // includes document id (optional)
typedef vector<SentencePair> Corpus;

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

template <class rnn_t>
int main_body(variables_map vm);

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);

    // command line processing
    variables_map vm; 
    options_description opts("Allowed options");
    opts.add_options()
        ("help", "print help message")
        ("config,c", value<string>(), "config file specifying additional command line options")
        ("train,t", value<string>(), "file containing training sentences, with "
            "each line consisting of source ||| target.")
        ("devel,d", value<string>(), "file containing development sentences.")
        ("test,T", value<string>(), "file containing testing sentences")
        ("rescore,r", "rescore (source, target) pairs in testing, default: translate source only")
        ("beam,b", value<int>()->default_value(0), "size of beam in decoding; 0=greedy")
        ("kbest,K", value<string>(), "test on kbest inputs using mononlingual Markov model")
        ("initialise,i", value<string>(), "load initial parameters from file")
        ("parameters,p", value<string>(), "save best parameters to this file")
        ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("align,a", value<int>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
        ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
        ("topk,k", value<int>()->default_value(100), "use <num> top kbest entries, used with --kbest")
        ("epochs,e", value<int>()->default_value(50), "maximum number of training epochs")
        ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
        ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
        ("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
        ("giza", "use GIZA++ style features in attentional components (corresponds to all of the 'gz' options)")
        ("gz-position", "use GIZA++ positional index features")
        ("gz-markov", "use GIZA++ markov context features")
        ("gz-fertility", "use GIZA++ fertility type features")
        ("curriculum", "use 'curriculum' style learning, focusing on easy problems in earlier epochs")
        ("swap", "swap roles of source and target, i.e., learn p(source|target)")
        ("document,D", "use previous sentence as document context; requires document id prefix in input files")
        ("coverage,C", value<float>()->default_value(0.0f), "impose alignment coverage penalty in training, with given coefficient")
        ("fertility,f", "learn Normal model of word fertility values")
        ("fert-stats,F", "display computed fertility values on the development set")
        ("display", "just display alignments instead of training or decoding")
        ("verbose,v", "be extremely chatty")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && !(vm.count("test") == 0 || vm.count("kbest") == 0 || vm.count("fert-stats") == 0))) {
        cout << opts << "\n";
        return 1;
    }

    if (vm.count("lstm"))
	return main_body<LSTMBuilder>(vm);
    else if (vm.count("gru"))
	return main_body<GRUBuilder>(vm);
    else
	return main_body<SimpleRNNBuilder>(vm);
}

void initialise(Model &model, const string &filename);

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, 
        bool doco, float coverage, bool display, bool fert);

template <class AM_t> void test_rescore(Model &model, AM_t &am, Corpus &testing, bool doco);
template <class AM_t> void test_decode(Model &model, AM_t &am, std::string test_file, bool doco, int beam);
template <class AM_t> void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k);
template <class AM_t> void fert_stats(Model &model, AM_t &am, Corpus &devel, bool global_fert);

const Sentence* context(const Corpus &corpus, unsigned i);

Corpus read_corpus(const string &filename, bool doco);
std::vector<int> ReadNumberedSentence(const std::string& line, Dict* sd, std::vector<int> &ids);
void ReadNumberedSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int> &ids);

template <class rnn_t>
int main_body(variables_map vm)
{
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
    kTGT_SOS = td.Convert("<s>");
    kTGT_EOS = td.Convert("</s>");
    verbose = vm.count("verbose");

    LAYERS = vm["layers"].as<int>(); 
    ALIGN_DIM = vm["align"].as<int>(); 
    HIDDEN_DIM = vm["hidden"].as<int>(); 
    bool bidir = vm.count("bidirectional");
    bool giza = vm.count("giza");
    bool giza_pos = giza || vm.count("gz-position");
    bool giza_markov = giza || vm.count("gz-markov");
    bool giza_fert = giza || vm.count("gz-fertility");
    bool fert = vm.count("fertility");
    bool swap = vm.count("swap");
    bool doco = vm.count("document");
    string flavour = "RNN";
    if (vm.count("lstm"))	flavour = "LSTM";
    else if (vm.count("gru"))	flavour = "GRU";

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel, testing;
    string line;
    cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
    training = read_corpus(vm["train"].as<string>(), doco);
    sd.Freeze(); // no new word types allowed
    td.Freeze(); // no new word types allowed
    
    SRC_VOCAB_SIZE = sd.size();
    TGT_VOCAB_SIZE = td.size();

    if (vm.count("devel")) {
	cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
	devel = read_corpus(vm["devel"].as<string>(), doco);
    }

    if (vm.count("test") && vm.count("rescore")) {
        // otherwise "test" file is assumed just to contain source language strings
        cerr << "Reading test examples from " << vm["test"].as<string>() << endl;
        testing = read_corpus(vm["test"].as<string>(), doco);
    }

    if (swap) {
	cerr << "Swapping role of source and target\n";
        std::swap(sd, td);
        std::swap(kSRC_SOS, kTGT_SOS);
        std::swap(kSRC_EOS, kTGT_EOS);
        std::swap(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE);
        for (auto &sent: training)
            std::swap(get<0>(sent), get<1>(sent));
        for (auto &sent: devel)
            std::swap(get<0>(sent), get<1>(sent));
        for (auto &sent: testing)
            std::swap(get<0>(sent), get<1>(sent));
    }

    string fname;
    if (vm.count("parameters"))
    	fname = vm["parameters"].as<string>();
    else if (vm.count("initialise"))
    	fname = vm["initialise"].as<string>();
    else {
		ostringstream os;
		os << "am"
			<< '_' << LAYERS
			<< '_' << HIDDEN_DIM
			<< '_' << ALIGN_DIM
			<< '_' << flavour
			<< "_b" << bidir
			<< "_g" << (int)giza_pos << (int)giza_markov << (int)giza_fert
			<< "_d" << doco
			<< "-pid" << getpid() << ".params";
		fname = os.str();
	}

	cerr << "Parameters will be written to: " << fname << endl;

	Model model;
    //bool use_momentum = false;
    Trainer* sgd = nullptr;
    //if (use_momentum)
        //sgd = new MomentumSGDTrainer(&model);
    //else
        sgd = new SimpleSGDTrainer(&model);
    //sgd = new AdadeltaTrainer(&model);

   cerr << "%% Using " << flavour << " recurrent units" << endl;
   AttentionalModel<rnn_t> am(&model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
	    LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza_pos, giza_markov, giza_fert, doco, !fert);

   bool add_fer = false;
   if (vm.count("test") && vm.count("rescore"))
   {
	   am.add_fertility_params(&model, HIDDEN_DIM, bidir);
	   add_fer = true;
   }

   if (vm.count("initialise")) initialise(model, vm["initialise"].as<string>());

   if (fert && !add_fer) am.add_fertility_params(&model, HIDDEN_DIM, bidir);

   if (!vm.count("test") && !vm.count("kbest") && !vm.count("fert-stats"))
    	train(model, am, training, devel, *sgd, fname, vm.count("curriculum"),
                vm["epochs"].as<int>(), doco, vm["coverage"].as<float>(), vm.count("display"),
                fert);
    else if (vm.count("kbest"))
    	test_kbest_arcs(model, am, vm["kbest"].as<string>(), vm["topk"].as<int>());
    else if (vm.count("test")) {
        if (vm.count("rescore"))
            test_rescore(model, am, testing, doco);
        else // test
            test_decode(model, am, vm["test"].as<string>(), doco, vm["beam"].as<int>());
    }
    else if (vm.count("fert-stats"))
        fert_stats(model, am, devel, vm.count("fertility"));

    delete sgd;

    //cnn::Free();

    return EXIT_SUCCESS;
}

template <class AM_t>
void test_rescore(Model &model, AM_t &am, Corpus &testing, bool doco)
{
    double tloss = 0;
    int tchars = 0;
    int lno = 0;

    Sentence ssent, tsent;
    int docid;
    for (unsigned i = 0; i < testing.size(); ++i) {
        tie(ssent, tsent, docid) = testing[i];

	ComputationGraph cg;
        am.BuildGraph(ssent, tsent, cg, nullptr, (doco) ? context(testing, i) : nullptr);

	double loss = as_scalar(cg.forward());
        cout << i << " |||";
	for (auto &w: ssent)
	    cout << " " << sd.Convert(w);
	cout << " |||";
	for (auto &w: tsent)
	    cout << " " << td.Convert(w);
	cout << " ||| " << (loss / (tsent.size()-1)) << endl;
	tloss += loss;
	tchars += tsent.size() - 1;

	if (verbose)
	    cerr << "chug " << lno++ << "\r" << flush;
    }

    cerr << "\n***TEST E = " << (tloss / tchars) << " ppl=" << exp(tloss / tchars) << ' ';
    return;
}

template <class AM_t>
void test_decode(Model &model, AM_t &am, string test_file, bool doco, int beam)
{
    double tloss = 0;
    int tchars = 0;
    int lno = 0;

    cerr << "Reading test examples from " << test_file << endl;
    ifstream in(test_file);
    assert(in);
    string line;
    Sentence last_source;
    Sentence source;
    int last_docid = -1;
    while (getline(in, line)) {
        vector<int> num;
        if (doco)
            source = ReadNumberedSentence(line, &sd, num);
        else 
            source = ReadSentence(line, &sd);

	if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
	    abort();
	}

	ComputationGraph cg;
        std::vector<int> target;

        if (beam > 0)
            target = am.beam_decode(source, cg, beam, td, (doco && num[0] == last_docid) ? &last_source : nullptr);
        else
            target = am.greedy_decode(source, cg, td, (doco && num[0] == last_docid) ? &last_source : nullptr);

        bool first = true;
	for (auto &w: target) {
            if (!first) cout << " ";
            cout << td.Convert(w);
            first = false;
        }
	cout << endl;

	if (verbose)
	    cerr << "chug " << lno++ << "\r" << flush;

        if (doco) {
            last_source = source;
            last_docid = num[0];
        }
    }
    return;
}

template <class AM_t>
void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k)
{
    // only suitable for monolingual setting, of predicting a sentence given preceeding sentence
    cerr << "Reading test examples from " << test_file << endl;
    unsigned lno = 0;
    ifstream in(test_file);
    assert(in);
    string line, last_id;
    const std::string sep = "|||";
    vector<SentencePair> items, last_items;
    last_items.push_back(SentencePair(Sentence({ kSRC_SOS, kSRC_EOS }), Sentence({ kTGT_SOS, kTGT_EOS }), -1));
    unsigned snum = 0;
    unsigned count = 0;

    auto process = [&am, &snum, &last_items, &items, &count]() {
        for (unsigned i = 0; i < last_items.size(); ++i) {
            ComputationGraph cg;
            auto &source = get<0>(last_items[i]);
            am.start_new_instance(source, cg);

            for (unsigned j = 0; j < items.size(); ++j) {
                std::vector<Expression> errs;
                auto &target = get<1>(items[j]);
                const unsigned tlen = target.size() - 1;
                for (unsigned t = 0; t < tlen; ++t) {
                    Expression i_r_t = am.add_input(target[t], t, cg);
                    Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
                    errs.push_back(i_err);
                }
                Expression i_nerr = sum(errs);
                double loss = as_scalar(cg.incremental_forward());

                //cout << last_last_id << ":" << last_id << " |||";
                //for (auto &w: source) cout << " " << sd.Convert(w);
                //cout << " |||";
                //for (auto &w: target) cout << " " << td.Convert(w);
                //cout << " ||| " << loss << "\n";
                cout << snum << '\t' << i << '\t' << j << '\t' << loss << '\n';
                ++count;
            }
        }
    };

    while (getline(in, line)) {
	Sentence source, target;

	istringstream in(line);
	string id, word;
	in >> id >> word;
	assert(word == sep);
	while(in) {
	    in >> word;
	    if (word.empty() || word == sep) break;
	    source.push_back(sd.Convert(word));
	    target.push_back(td.Convert(word));
	}

	if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
		(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
	    abort();
	}

	if (id != last_id && !items.empty()) {
	    if (items.size() > top_k)
		items.resize(top_k);

            process();

	    last_items = items;
	    last_id = id;
	    items.clear();
            snum++;

	    if (verbose)
		cerr << "chug " << lno++ << " [" << count << " pairs]\r" << flush;
	}

	last_id = id;
	items.push_back(SentencePair(source, target, -1));
    }
    
    if (!items.empty())
        process();

    return;
}

template <class AM_t> 
void fert_stats(Model &model, AM_t &am, Corpus &devel, bool global_fert)
{
    Sentence ssent, tsent;
    int docid;

    if (global_fert) {
        std::cout << "==== FERTILITY ESTIMATES ====\n";
        for (unsigned i = 0; i < devel.size(); ++i) {
            tie(ssent, tsent, docid) = devel[i];
            std::cout << "=== sentence " << i << " (" << docid << ") ===\n";
            am.display_fertility(ssent, sd);
        }
    }

    std::cout << "==== EMPIRICAL FERTILITY VALUES ====\n";
    for (unsigned i = 0; i < devel.size(); ++i) {
        tie(ssent, tsent, docid) = devel[i];
        std::cout << "=== sentence " << i << " (" << docid << ") ===\n";
        am.display_empirical_fertility(ssent, tsent, sd);
    }
}

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, 
        bool doco, float coverage, bool display, bool fert)
{
    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500; 
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    vector<vector<unsigned>> order_by_length; 
    const unsigned curriculum_steps = 10;
    if (curriculum) {
	// simple form of curriculum learning: for the first K epochs, use only
	// the shortest examples from the training set. E.g., K=10, then in
	// epoch 0 using the first decile, epoch 1 use the first & second
	// deciles etc. up to the full dataset in k >= 9.
	multimap<size_t, unsigned> lengths;
	for (unsigned i = 0; i < training.size(); ++i) 
	    lengths.insert(make_pair(get<0>(training[i]).size(), i));

	order_by_length.resize(curriculum_steps);
	unsigned i = 0;
	for (auto& landi: lengths) {
	    for (unsigned k = i * curriculum_steps / lengths.size(); k < curriculum_steps; ++k)  
		order_by_length[k].push_back(landi.second);
	    ++i;
	}
    }

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;
    Sentence ssent, tsent;
    int docid;

    // FIXME: move this into sep function
    if (display) {
        // display the alignments
        //
        for (unsigned i = 0; i < devel.size(); ++i) {
            tie(ssent, tsent, docid) = devel[i];
            ComputationGraph cg;
            Expression alignment;
            am.BuildGraph(ssent, tsent, cg, &alignment, (doco) ? context(devel, i) : nullptr);
            cg.forward();

            cout << "\n====== SENTENCE " << i << " =========\n";
            am.display_ascii(ssent, tsent, cg, alignment, sd, td);
            cout << "\n";

            am.display_tikz(ssent, tsent, cg, alignment, sd, td);
            cout << "\n";
        }

        return;
    }

#if 0
        if (true) {
            double dloss = 0;
            int dchars = 0;
            for (unsigned i = 0; i < devel.size(); ++i) {
                tie(ssent, tsent, docid) = devel[i];
                ComputationGraph cg;
                am.BuildGraph(ssent, tsent, cg, nullptr, (doco) ? context(devel, i) : nullptr);
                dloss += as_scalar(cg.forward());
                dchars += tsent.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(out_file);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
#endif

    while (sgd.epoch < max_epochs) {
        Timer iteration("completed in");
        double loss = 0;
        double penalty = 0;
        double loss_fert = 0;
        unsigned words_src = 0;
        unsigned words_tgt = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; } else { sgd.update_epoch(); }

		if (curriculum && epoch < order_by_length.size()) {
		    order = order_by_length[epoch++];
		    cerr << "Curriculum learning, with " << order.size() << " examples\n";
		} 
	    }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
	    }

	    if (verbose && iter+1 == report_every_i) {
		tie(ssent, tsent, docid) = training[order[si % order.size()]];
		ComputationGraph cg;
                cerr << "\nDecoding source, greedy Viterbi: ";
                am.greedy_decode(ssent, cg, td, (doco) ? context(training, order[si % order.size()]) : nullptr);

                cerr << "\nDecoding source, sampling: ";
                am.sample(ssent, cg, td, (doco) ? context(training, order[si % order.size()]) : nullptr);
	    }

            // build graph for this instance
	    tie(ssent, tsent, docid) = training[order[si % order.size()]];
	    ComputationGraph cg;
            words_src += ssent.size() - 1;
            words_tgt += tsent.size() - 1;
            ++si;
            Expression alignment, coverage_penalty, fertility_nll;
            Expression xent = am.BuildGraph(ssent, tsent, cg, &alignment, 
                    (doco) ? context(training, order[si % order.size()]) : nullptr, 
                    (coverage > 0) ? &coverage_penalty : nullptr,
                    (fert) ? &fertility_nll : nullptr);

            Expression objective = xent;
            if (coverage > 0) 
                objective = objective + coverage * coverage_penalty;
            if (fert) 
                objective = objective + fertility_nll;

            // perform forward computation for aggregate objective
            cg.forward();

            // grab the parts of the objective
            loss += as_scalar(cg.get_value(xent.i));
            if (coverage < 0) 
                penalty += as_scalar(cg.get_value(coverage_penalty.i));
            if (fert) 
                loss_fert += as_scalar(cg.get_value(fertility_nll.i));
            
            cg.backward();
            sgd.update();
            ++lines;

	    if (verbose) {
		cerr << "chug " << iter << "\r" << flush;
		if (iter+1 == report_every_i) {
		    // display the alignment
		    am.display_ascii(ssent, tsent, cg, alignment, sd, td);
                    cout << "\n";
		    am.display_tikz(ssent, tsent, cg, alignment, sd, td);
                    cout << "\n";
		}
	    }
        }
        sgd.status();

        //loss -= coverage * penalty - loss_fert;
        cerr << " E = " << (loss / words_tgt) << " ppl=" << exp(loss / words_tgt) << ' ';
        if (coverage > 0) 
            cerr << "cover=" << penalty/words_src << ' ';
        if (fert)
            cerr << "fert_ppl=" << exp(loss_fert / words_src) << ' ';

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (unsigned i = 0; i < devel.size(); ++i) {
                tie(ssent, tsent, docid) = devel[i];
                ComputationGraph cg;
                am.BuildGraph(ssent, tsent, cg, nullptr, (doco) ? context(devel, i) : nullptr, nullptr, nullptr);
                dloss += as_scalar(cg.forward());
                dchars += tsent.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(out_file, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            //else{
            //	sgd.eta *= 0.5;
            //}
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
    }
}

Corpus read_corpus(const string &filename, bool doco)
{
    ifstream in(filename);
    assert(in);
    Corpus corpus;
    string line;
    int lc = 0, stoks = 0, ttoks = 0;
    vector<int> identifiers({ -1 });
    while (getline(in, line)) {
        ++lc;
        Sentence source, target;
        if (doco) 
            ReadNumberedSentencePair(line, &source, &sd, &target, &td, identifiers);
        else
            ReadSentencePair(line, &source, &sd, &target, &td);
        corpus.push_back(SentencePair(source, target, identifiers[0]));
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

std::vector<int> ReadNumberedSentence(const std::string& line, Dict* sd, vector<int> &identifiers) {
    std::istringstream in(line);
    std::string word;
    std::vector<int> res;
    std::string sep = "|||";
    if (in) {
        identifiers.clear();
        while (in >> word) {
            if (!in || word.empty()) break;
            if (word == sep) break;
            identifiers.push_back(atoi(word.c_str()));
        }
    }

    while(in) {
        in >> word;
        if (!in || word.empty()) break;
        res.push_back(sd->Convert(word));
    }
    return res;
}


void ReadNumberedSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, vector<int> &identifiers) 
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd;
    std::vector<int>* v = s; 

    if (in) {
        identifiers.clear();
        while (in >> word) {
            if (!in || word.empty()) break;
            if (word == sep) break;
            identifiers.push_back(atoi(word.c_str()));
        }
    }

    while(in) {
        in >> word;
        if (!in) break;
        if (word == sep) { d = td; v = t; continue; }
        v->push_back(d->Convert(word));
    }
}

void initialise(Model &model, const string &filename)
{
    cerr << "Initialising model parameters from file: " << filename << endl;
    ifstream in(filename, ifstream::in);
    boost::archive::text_iarchive ia(in);
    ia >> model;
}

const Sentence* context(const Corpus &corpus, unsigned i)
{
    if (i > 0) {
        int docid = get<2>(corpus.at(i));
        int prev_docid = get<2>(corpus.at(i-1));
        if (docid == prev_docid) 
            return &get<0>(corpus.at(i-1));
    } 
    return nullptr;
}
