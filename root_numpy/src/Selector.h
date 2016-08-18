#ifndef __SELECTOR_H
#define __SELECTOR_H

#include <vector>
#include <TTreeFormula.h>


class Selector
{
    public:

    Selector(TTreeFormula* selection):
        selection(selection) {}

    ~Selector()
    {
        // The TreeChain owns the selection TTreeFormula
    }

    void Update()
    {
        // TreeChain calls this method
        int num_elements = selection->GetNdata();
        num_selected = 0;
        selected.clear();
        selected.assign(num_elements, false);
        for (int i = 0; i < num_elements; ++i)
        {
            if (selection->EvalInstance(i) != 0)
            {
                selected[i] = true;
                ++num_selected;
            }
        }
    }

    TTreeFormula* selection;
    std::vector<bool> selected;
    unsigned int num_selected;
};

#endif
