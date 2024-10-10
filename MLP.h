#pragma once
#define LEARNING_RATE0
class CMLP
{
public:
	CMLP();
	~CMLP();

	int m_iNuminNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;
	int m_iNumTotalLayer;
	int* m_NumNodes;

	double*** m_Weight;
	double** m_NodeOut;

	double* pInValue, * pOutValue;
	double* pCorrectOutValue;

	double** m_ErrorGradient;


	bool Create(int InNode, int* pHiddenNode, int OutNode, int NumHiddenLayer);

private:
	void InitW();
	double ActivationFunc(double weightsum);

public:
	void Forward();
	void BackPopagationLearning();
};
