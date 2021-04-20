class InfixToPostfix:
	findvalue = {}
	def __init__(self):
		self.findvalue = {'+': 1, '-': 1, '*': 2, '/':2}
		
	def operator(self, c):
		return self.findvalue[c]
	
	def isNumericValue(self, c):
		if c >= '0' and c <= '9':
			return True
		return False
		
	def infixtopostfix(self, s):
		stack = []
		result = ""
		
		for char in s:
			
			if self.isNumericValue(char):
				#push operands (Number) into the stack
				result = result + char
				
			elif char == '(':
				#push if it is opening bracket
				stack.append(char)
				
			elif char == ')':
				while len(stack) != 0 and stack[len(stack)-1] != '(':
					result = result + stack.pop()
				stack.pop()
					
			else:
				while len(stack) != 0 and stack[len(stack)-1] != '(' and self.operator(stack[len(stack)-1]) >= self.operator(char):
					result = result + stack.pop()
				stack.append(char)
				
		while len(stack) != 0:
			result = result + stack.pop();
			
		return result
			
Itop = InfixToPostfix()
