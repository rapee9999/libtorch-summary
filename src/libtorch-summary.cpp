//
// Mimick Python's torchsummary
//

#include "libtorch-summary.h"


// set string from any variable
#define SET_STR(srcVal, dstStr)	\
	buffer.str("");				\
	buffer << srcVal;			\
	dstStr = buffer.str();		\


// deprecated : change to ModuleInfo because store only string for display only
struct ModuleData {
	ModuleData(std::shared_ptr<torch::jit::Module> module, std::string name) :
		module(module), name(name) {};
	std::shared_ptr<torch::jit::Module> module; // pointer to module
	std::string name; // module name
	std::vector<torch::IntArrayRef> shapeParames; // total dimesion of all parameters under the module
	std::vector<caffe2::TypeMeta> dtypeParams; // data types of all parameters under the module
	int64_t numParams; // number of all parameters under the module
	int64_t sizeParams; // byte size of all parameters under the module
	torch::IntArrayRef inputShape; // dimension of input tensor fed to the module
	caffe2::TypeMeta inputDtype; // data type of input tensor fed to the module
	torch::IntArrayRef outputShape; // dimension of output tensor from the module
	caffe2::TypeMeta outputDtype; // data type of output tensor from the module
	std::vector<ModuleData*> childModules; // sub-modules under the module
};


// store module info by string
struct ModuleInfo {
	ModuleInfo(std::shared_ptr<torch::jit::Module> module, std::string name) :
		module(module), name(name) {};
	std::shared_ptr<torch::jit::Module> module; // pointer to module
	std::string name; // module name
	std::string shapeParames; // total dimesion of all parameters under the module
	std::string dtypeParams; // data types of all parameters under the module
	std::string numParams; // number of all parameters under the module
	std::string sizeParams; // byte size of all parameters under the module
	std::string inputShape; // dimension of input tensor fed to the module
	std::string inputDtype; // data type of input tensor fed to the module
	std::string outputShape; // dimension of output tensor from the module
	std::string outputDtype; // data type of output tensor from the module
	std::vector<ModuleInfo*> childModules; // sub-modules under the module
};


/*
 * Get module information from module properties and inferencing.
 *
 * @param pModuleInfo Pointer to a ModuleInfo object.
 * @param tInput Input tensor.
 * @return Output tensor
 */
torch::Tensor getModuleInfo(ModuleInfo* pModuleInfo, torch::Tensor tInput)
{
	// list of parameter tensors (each member in the last is counted as one parameter which is one tensor)
	std::shared_ptr<torch::jit::Module> shPtrModule = pModuleInfo->module;
	torch::jit::parameter_list listParams = pModuleInfo->module->parameters(); // get a list of parameters
	
	// iterator to each parameter
	torch::jit::slot_iterator_impl<torch::jit::detail::ParameterPolicy> iterParam = listParams.begin();

	// initialize store variables
	int numParams = 0; // total number of parameters
	int sizeParams = 0; // total size of parameters
	std::vector<torch::IntArrayRef> shapeParames;
	std::vector<caffe2::TypeMeta> dtypeParams;
	torch::IntArrayRef inputShape;
	caffe2::TypeMeta inputDtype;
	torch::IntArrayRef outputShape;
	caffe2::TypeMeta outputDtype;

	// loop per one parameter tensor
	for (; iterParam != listParams.end(); iterParam++) {
		torch::Tensor tParam = *iterParam; // get tensor from iterator
		torch::IntArrayRef arrParam = tParam.sizes(); // get an array of tensor dimension size
		caffe2::TypeMeta dtypeParam = tParam.dtype();
		int64_t iParamMulti = 1; // multiplied number of parameter
		torch::IntArrayRef::iterator iterDim = arrParam.begin(); // get dimension iterator

		// loop per tensor dimension to multiply dimensions to get total parameters (points) in one tensor
		for (; iterDim < arrParam.end(); iterDim++) iParamMulti *= *iterDim;

		// store info
		shapeParames.push_back(arrParam);
		dtypeParams.push_back(dtypeParam);
		numParams += iParamMulti; // sum number of parameters in each tensor
		sizeParams += (iParamMulti * dtypeParam.itemsize()); // sum size of parameters in each tensor
	}

	// feed forward
	torch::Tensor tOutput;
	if (tInput.size(0) > 0) {
		try {
			// forward
			std::vector<torch::jit::IValue> vInput;
			vInput.push_back(tInput);
			tOutput = shPtrModule->forward(vInput).toTensor();

			// store info
			inputShape = tInput.sizes();
			inputDtype = tInput.dtype();
			outputShape = tOutput.sizes();
			outputDtype = tOutput.dtype();
		}
		catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.what() << std::endl;
			throw e;
		}
	}
	else {
		tOutput = torch::Tensor();
	}

	// store info
	std::stringstream buffer;
	SET_STR(numParams, pModuleInfo->numParams)
	SET_STR(sizeParams, pModuleInfo->sizeParams)
	SET_STR(shapeParames, pModuleInfo->shapeParames)
	SET_STR(dtypeParams, pModuleInfo->dtypeParams)
	SET_STR(inputShape, pModuleInfo->inputShape)
	SET_STR(inputDtype, pModuleInfo->inputDtype)
	SET_STR(outputShape, pModuleInfo->outputShape)
	SET_STR(outputDtype, pModuleInfo->outputDtype)

	// initialise to sub-modules
	torch::jit::module_list listSubModules = pModuleInfo->module->children(); // list of immediate sub-modules	
	// iterator to each sub-module
	torch::jit::slot_iterator_impl<torch::jit::detail::ModulePolicy> iterSubModule = listSubModules.begin();
	// list of immediat submoeuls' name
	torch::jit::named_module_list listSubModuleNames = pModuleInfo->module->named_children();
	// iterator to each sub-module's name
	torch::jit::slot_iterator_impl<
		torch::jit::detail::NamedPolicy<
			torch::jit::detail::ModulePolicy
			>
		> iterSubModuleName = listSubModuleNames.begin();

	// loop per sub-module
	while (iterSubModule != listSubModules.end()) {
		ModuleInfo* pSubModuleInfo = new ModuleInfo(
			std::make_shared<torch::jit::Module>(*iterSubModule), 
			(*iterSubModuleName).name );
		pModuleInfo->childModules.push_back(pSubModuleInfo); // add this module info into parent module info
		tInput = getModuleInfo(pSubModuleInfo, tInput); // recursive into children modules

		iterSubModule++;
		iterSubModuleName++;
	}

	return tOutput;
};


// Set text in cell in model summary table
class TextCell {
	public:
		std::string sepStr; // separator string between cell
		int sepWidth; // number of charactors in a separator
		int cellWidth; // number of charactors in a cell

		TextCell(int cellWidth, std::string sepStr) : cellWidth(cellWidth), sepStr(sepStr) {
			sepWidth = sepStr.length();
		}

		/*
		 * Fit the given string in a cell.
		 *
		 * @param str String to be in a cell.
		 * @return std::string String fitted to a cell.
		 */
		std::string fitText2Cell(std::string str, bool appendSep = true) {
			int spaceNum = cellWidth - str.length();
			if (spaceNum >= 0) {
				str.append(std::string(spaceNum, ' '));
			}
			else {
				str = str.substr(0, cellWidth - 2) + "..";
			}

			if (appendSep) {
				str.append(sepStr);
			}

			return str;
		}
};


/*
 * Print headers.
 *
 * @param pTextCell TextCell object to print cell in model summary table
 */
void printHeader(TextCell * pTextCell) {
	TextCell mergeCell = TextCell((pTextCell->cellWidth * 3) + (pTextCell->sepStr.length() * 2), " | ");
	std::cout 		
		<< pTextCell->fitText2Cell("Module Name") 	
		<< mergeCell.fitText2Cell("Parameters")
		<< pTextCell->fitText2Cell("Output Dimension", false)
	<< std::endl 	
		<< pTextCell->fitText2Cell("") 			
		<< pTextCell->fitText2Cell("Dimension")		
		<< pTextCell->fitText2Cell("Number") 
		<< pTextCell->fitText2Cell("Size (byte)") 
		<< pTextCell->fitText2Cell("", false)
	<< std::endl 	
		<< std::string((pTextCell->cellWidth * 5) + (pTextCell->sepStr.length() * 4), '-')
	<< std::endl;
}


/*
 * Print input.
 *
 * @param pModuleInfo Pointer to a ModuleInfo object.
 * @param pTextCell TextCell object to print cell in model summary table
 */
void printInput(ModuleInfo * pModuleInfo, TextCell * pTextCell) {
	std::cout 		
		<< pTextCell->fitText2Cell("Input")
		<< pTextCell->fitText2Cell("")
		<< pTextCell->fitText2Cell("")
		<< pTextCell->fitText2Cell("")
		<< pTextCell->fitText2Cell(pModuleInfo->inputShape, false)
	<< std::endl;	
}


/*
 * Print layer.
 *
 * @param pModuleInfo Pointer to a ModuleInfo object.
 * @param strPrefix String before printing module information.
 * @param pTextCell TextCell object to print cell in model summary table
 */
void printLayer(ModuleInfo* pModuleInfo, std::string strPrefix, TextCell * pTextCell) {
	std::string strDisplayName = (strPrefix + pModuleInfo->name);

	std::cout 		
		<< pTextCell->fitText2Cell(strDisplayName)
		<< pTextCell->fitText2Cell(pModuleInfo->shapeParames)
		<< pTextCell->fitText2Cell(pModuleInfo->numParams)
		<< pTextCell->fitText2Cell(pModuleInfo->sizeParams)
		<< pTextCell->fitText2Cell(pModuleInfo->outputShape, false)
	<< std::endl;

	std::vector<ModuleInfo*> vecSubModules = pModuleInfo->childModules;
	std::vector<ModuleInfo*>::iterator iterSubModule = vecSubModules.begin();
	for (; iterSubModule != vecSubModules.end(); iterSubModule++) {
		printLayer(*iterSubModule, strPrefix.substr(0,2) + " |- ", pTextCell);
	}
}


/*
 * Print module information.
 *
 * @param pModuleInfo Pointer to a ModuleInfo object.
 * @param cellWidth Cell width
 * @param sepStr Separator string between cells
 */
void printModuleInfo(ModuleInfo* pModuleInfo, int cellWidth, std::string sepStr) {
	TextCell textCell = TextCell(cellWidth, sepStr); 
	printHeader(&textCell);
	printInput(pModuleInfo, &textCell);
	printLayer(pModuleInfo, "", &textCell);
};


void torchsummary::summary(
	std::shared_ptr<torch::jit::Module> shPtrModule, 
	std::vector<int64_t> vecInputShape, 
	std::string strModuleName, 
	int cellWidth )
{
	// initialise model
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	shPtrModule->to(device);
	shPtrModule->eval();

	// initialise input tensor
	torch::Tensor tInput;
	if (vecInputShape.size() > 0) {
		tInput = torch::rand(vecInputShape).to(device);
	}
	else {
		tInput = torch::Tensor();
	}

	// get model info
	ModuleInfo* pModuleInfo = new ModuleInfo(shPtrModule, strModuleName);
	getModuleInfo(pModuleInfo, tInput);
	printModuleInfo(pModuleInfo, cellWidth, " | ");
}
