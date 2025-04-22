class IndividualEval:
    def __init__(self, input: str, actual_output: str, expected_output: str, retrieval_context: str, index: int):
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.retrieval_context = retrieval_context
        self.index = index

    def evaluate(self):
        output_score = 0
        retrieval_score = 0
        
        match self.index:
            case 1:
                output_score = 1 if any(x in self.actual_output for x in ["7,5", "7,50", "7.5", "7.50"]) else 0
                retrieval_score = 1 if "7,50" in self.retrieval_context else 0
            
            case 2:                
                output_score = 1 if "25" in self.actual_output else 0
                retrieval_score = 1 if "25" in self.actual_output else 0

            case 3:
                output_score = 1 if any(x in self.actual_output for x in ["1600", "1.600", "1000", "1.000"]) else 0
                retrieval_score = 1 if any(x in self.actual_output for x in ["§ 7c", "§7c"]) else 0

            case 4:
                output_score = 1 if "20" in self.actual_output else 0
                retrieval_score = 1 if any(x in self.actual_output for x in ["§ 83", "§83"]) else 0

            case 5:
                output_score = 1 if "26" in self.actual_output else 0
                retrieval_score = 1 if any(x in self.actual_output for x in ["§ 70a", "§70a"]) else 0


            case 6:
                output_score = 1 if "2" in self.actual_output else 0
                retrieval_score = 1 if any(x in self.actual_output for x in ["§ 79", "§79"]) else 0


            case 7:
                output_score = 1 if any(x in self.actual_output for x in ["Nein", "nicht"]) else 0
                retrieval_score = 1 if any(x in self.actual_output for x in ["§ 62a", "§62a"]) else 0
                
 
        return {
            "index": self.index,
            "output_score": output_score,
            "retrieval_score": retrieval_score
        }

# # Example usage
# indi = IndividualEval(
#     input="1. Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?",
#     actual_output="1. Ein Gebäude in Bauklasse I darf maximal 7,50 Meter hoch sein.",
#     expected_output="1. Ein Gebäude in Bauklasse I darf maximal 7,50 Meter hoch sein.",
#     retrieval_context="1. Ein Gebäude in Bauklasse I darf maximal 7,50 Meter hoch sein.",
#     index=8
# )

# result = indi.evaluate()
# print(result)
