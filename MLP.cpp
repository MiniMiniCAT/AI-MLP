#include "MLP.h"
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


CMLP::CMLP()
{
	int layer;

	m_iNuminNodes = 0;
	m_iNumOutNodes = 0;

	m_NumNodes = NULL;
	m_NodeOut = NULL;

	m_Weight = NULL;
	m_ErrorGradient = NULL;

	pInValue = NULL;
	pOutValue = NULL;
	pCorrectOutValue = NULL;
}

CMLP::~CMLP()
{
	int layer, snode;

	if (m_NodeOut != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer + 1); layer++)
			free(m_NodeOut[layer]);
		free(m_NodeOut);
	}

	if (m_Weight != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer - 1); layer++)
		{
			if (m_Weight[layer] != NULL)
			{
				for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
					free(m_Weight[layer][snode]);
				free(m_Weight[layer]);
			}
		}
	}

	if (m_ErrorGradient != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer); layer++)
			free(m_ErrorGradient[layer]);
		free(m_ErrorGradient);
	}

	if (m_NumNodes != NULL)
		free(m_NumNodes);
}

bool CMLP::Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer)
{
	int layer, snode, enode;

	m_iNuminNodes = InNode;
	m_iNumOutNodes = OutNode;
	m_iNumHiddenLayer = numHiddenLayer;
	m_iNumTotalLayer = numHiddenLayer + 2;

	m_NumNodes = (int*)malloc((m_iNumTotalLayer + 1) * sizeof(int));

	m_NumNodes[0] = m_iNuminNodes;
	for (layer = 0; layer < m_iNumHiddenLayer; layer++)
		m_NumNodes[1 + layer] = pHiddenNode[layer];
	m_NumNodes[m_iNumTotalLayer - 1] = m_iNumOutNodes;
	m_NumNodes[m_iNumTotalLayer] = m_iNumOutNodes;

	m_NodeOut = (double**)malloc((m_iNumTotalLayer + 1) * sizeof(double*));
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
		m_NodeOut[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));
	m_NodeOut[m_iNumTotalLayer] = (double*)malloc((m_NumNodes[m_iNumTotalLayer - 1] + 1) * sizeof(double));

	m_Weight = (double***)malloc((m_iNumTotalLayer - 1) * sizeof(double**));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		m_Weight[layer] = (double**)malloc((m_NumNodes[layer] + 1) * sizeof(double*));
		for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
			m_Weight[layer][snode] = (double*)malloc((m_NumNodes[layer + 1] + 1) * sizeof(double));
	}

	pInValue = m_NodeOut[0];
	pOutValue = m_NodeOut[m_iNumTotalLayer - 1];
	pCorrectOutValue = m_NodeOut[m_iNumTotalLayer];

	InitW();

	for (layer = 0; layer < m_iNumTotalLayer + 1; layer++)
	{	m_NodeOut[layer][0] = 1;	}
	return true;

}

void CMLP::InitW()
{
	int layer, snode, enode;

	srand(time(NULL));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (snode = 0; snode <= m_NumNodes[layer]; snode++)
		{
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
			{
				m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX - 0.5;
			}
		}
	}
}


double CMLP::ActivationFunc(double weightsum)
{	return 1.0 / (1.0 + exp(-weightsum));	}




void CMLP::Forward()
{
	int layer, snode, enode;
	double wsum;

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			wsum = 0.0;
			wsum += m_Weight[layer][0][enode] * 1;
			for (snode = 1; snode <= m_NumNodes[layer]; snode++)
			{
				wsum += m_Weight[layer][snode][enode] * m_NodeOut[layer][snode];
			}
			m_NodeOut[layer + 1][enode] = ActivationFunc(wsum);
		}
	}
}

void CMLP::BackPopagationLearning()
{
	int layer;
	if (m_ErrorGradient == NULL)
	{
		m_ErrorGradient = (double**)malloc((m_iNumTotalLayer) * sizeof(double*));
		for (layer = 0; layer < m_iNumTotalLayer; layer++)
			m_ErrorGradient[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));
	}

	int snode, enode, node;
	for (node = 1; node <= m_iNumOutNodes; node++)
	{
		m_ErrorGradient[m_iNumTotalLayer - 1][node] =
			(pCorrectOutValue[node] - m_NodeOut[m_iNumTotalLayer - 1][node])
			* m_NodeOut[m_iNumTotalLayer - 1][node] * (1 - m_NodeOut[m_iNumTotalLayer - 1][node]);
	}

	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
	{
		for (snode = 1; snode <= m_NumNodes[layer]; snode++)
		{
			m_ErrorGradient[layer][snode] = 0.0;
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
			{
				m_ErrorGradient[layer][snode] += (m_ErrorGradient[layer + 1][enode] * m_Weight[layer][snode][enode]);
			}
			m_ErrorGradient[layer][snode] *= m_NodeOut[layer][snode] * (1 - m_NodeOut[layer][snode]);
		}
	}
}