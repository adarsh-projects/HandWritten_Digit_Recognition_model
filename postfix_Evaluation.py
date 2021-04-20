class PostfixEvaluation:
	stack = []
	def __init__(self):
		self.stack = []
		
	def isDigit(self,char):
		if char >= '0' and char <= '9':
			return True
		return False
	
	def solve_postfix_expression(self, s):
		for char in s:
			if self.isDigit(char):
				c = int(char)
				self.stack.append(c)
			elif char == '+':
				self.stack.append( self.stack.pop() + self.stack.pop())
			elif char == '-':
				t1 = self.stack.pop()
				t2 = self.stack.pop()
				self.stack.append(t2-t1)
			elif char == '*':
				self.stack.append(self.stack.pop() * self.stack.pop())
			else:
				t1 = self.stack.pop()
				t2 = self.stack.pop()
				self.stack.append(t2/t1)
		
		return self.stack.pop()

pe = PostfixEvaluation()
