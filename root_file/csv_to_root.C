#include <fstream>
#include <iostream>
#include <sstream>

#include "TFile.h"
#include "TTree.h"

int csv_to_root() {
  // Open the file
  std::ifstream f("E:/ML_data/vt/data/slow_pions_evtgen_big.txt");

  // Create the output file
  TFile* file = new TFile("data.root", "RECREATE");
  file->cd();

  // Create the tree
  TTree* tree = new TTree("data", "data");

  // Create the variables
  float adc[81];
  float global_pos[3];
  float adc_total;

  // Create the branches
  tree->Branch("adc", &adc, "adc[81]/F");
  tree->Branch("pos", &global_pos, "global[3]/F");
  tree->Branch("adc_total", &adc_total, "adc_total/F");

  std::string line;
  while (std::getline(f, line)) {
    // empty adc, global_pos, adc_total
    for (int i = 0; i < 81; i++)
      adc[i] = 0.;
    for (int i = 0; i < 3; i++)
      global_pos[i] = 0.;
    adc_total = 0.;

    std::istringstream iss(line);
    std::string value;
    int i = 0;
    while (std::getline(iss, value, ' ')) {
      if (i == 1)
        adc_total = std::stof(value);
      else if (i == 83)
        global_pos[0] = std::stof(value);
      else if (i == 84)
        global_pos[1] = std::stof(value);
      else if (i == 85)
        global_pos[2] = std::stof(value);
      else
        adc[i - 2] = std::stof(value);
      i++;
    }
    tree->Fill();
  }
  // Write the tree
  tree->Write();

  // Close the file
  file->Close();

  return 0;
}