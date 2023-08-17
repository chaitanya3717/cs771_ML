import numpy as np
import math

#we have implemented three strategies: get_group_word, get_group_word_2, get_group_word_3
#currently, we have called get_group_word in process_node method of Node class at Line No. 107
#change the function call to get_group_word_2 or get_group_word_3 to check other strategies 


def my_fit( words, verbose = False ):
	dt = Tree( min_leaf_size = 1, max_depth = 15 )
	dt.fit( words, verbose )
	return dt

class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, words, verbose = False ):
		self.words = words
		self.root = Node( depth = 0, parent = None )
		if verbose:
			print( "root" )
			print( "└───", end = '' )
		# The root is trained with all the words
		self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, verbose = verbose )


class Node:
	# A node stores its own depth (root = depth 0), a link to its parent
	# A link to all the words as well as the words that reached that node
	# A dictionary is used to store the children of a non-leaf node.
	# Each child is paired with the response that selects that child.
	# A node also stores the query-response history that led to that node
	# Note: my_words_idx only stores indices and not the words themselves
	def __init__( self, depth, parent ):
		self.depth = depth
		self.parent = parent
		self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
		self.history = []
	
	# Each node must implement a get_query method that generates the
	# query that gets asked when we reach that node. Note that leaf nodes
	# also generate a query which is usually the final answer
	def get_query( self ):
		return self.query_idx
	
	# Each non-leaf node must implement a get_child method that takes a
	# response and selects one of the children based on that response
	def get_child( self, response ):
		# This case should not arise if things are working properly
		# Cannot return a child if I am a leaf so return myself as a default action
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			# This should ideally not happen. The node should ensure that all possibilities
			# are covered, e.g. by having a catch-all response. Fix the model if this happens
			# For now, hack things by modifying the response to one that exists in the dictionary
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	# Dummy leaf action -- just return the first word
	def process_leaf( self, my_words_idx, history ):
		return my_words_idx[0]
	
	def reveal( self, word, query ):
		# Find out the intersections between the query and the word
		mask = [ *( '_' * len( word ) ) ]
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ' '.join( mask )
	
	# Dummy node splitting action -- use a random word as query
	# Note that any word in the dictionary can be the query
	def process_node( self, all_words, my_words_idx, history,depth, verbose ):
		# For the root we do not ask any query -- Melbot simply gives us the length of the secret word
		if len( history ) == 0:
			query_idx = -1
			query = ""
		else:
				sub_dict=[all_words[idx] for idx in my_words_idx]
				query_idx = my_words_idx[get_group_word(sub_dict)] # call get_group_word_2, get_group_word_3 here to run other strategies.
				query = all_words[query_idx]
				

        
		
		split_dict = {}
		
		for  idx in my_words_idx:
			

				mask = self.reveal( all_words[ idx ], query )
				if mask not in split_dict:
					split_dict[ mask ] = []
			
				split_dict[ mask ].append( idx )
		
		if len( split_dict.items() ) < 2 and verbose:
			print( "Warning: did not make any meaningful split with this query!" )
		
		return ( query_idx, split_dict )
	
	def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str = "    ", verbose = False ):
		self.all_words = all_words
		self.my_words_idx = my_words_idx
		
		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth :
			self.is_leaf = True
			self.query_idx = self.process_leaf( self.my_words_idx, self.history )
			if verbose:
				print( '█' )
		else:
			self.is_leaf = False
			( self.query_idx, split_dict ) = self.process_node( self.all_words, self.my_words_idx, self.history,self.depth, verbose )
			
			if verbose:
				print( all_words[ self.query_idx ] )
			
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				if verbose:
					if i == len( split_dict ) - 1:
						print( fmt_str + "└───", end = '' )
						fmt_str += "    "
					else:
						print( fmt_str + "├───", end = '' )
						fmt_str += "│   "
				
				# Create a new child for every split
				self.children[ response ] = Node( depth = self.depth + 1, parent = self )
				history = self.history.copy()
				history.append( [ self.query_idx, response ] )
				self.children[ response ].history = history
				
				# Recursively train this child node
				self.children[ response ].fit( self.all_words, split, min_leaf_size, max_depth, fmt_str, verbose )


def get_group_word(words):
    """
    Returns the word that groups the maximum number of words using its mask
    """
    # Get the set of unique characters in the list of words
    char_set = set(''.join(words))
    
    # Create a dictionary to store the frequency of mask characters
    freq_dict = {}
    for char in char_set:
        freq_dict[char] = [0] * len(words[0])
    
    # Loop through all the words and update the frequency dictionary
    for word in words:
        
        for i, char in enumerate(word):
          
            freq_dict[char][i] += 1
    for char in char_set:
       for i in range(len(words[0])):
          freq_dict[char][i] /= len(words)
    # Find the word with the highest frequency of mask characters at the correct position
    max_word = ''
    max_count = 0
    ind=0
    for (j,word) in enumerate(words):
        
        count = sum([freq_dict[char][i] for i, char in enumerate(word)])
        if count > max_count:
            ind=j
            max_word = word
            max_count = count
    
    return ind
   # 0.20315787600002294 1089774.0 1.0 4.058834913876524


def get_group_word_2(words):
    score=[0] * len(words) 
    for ( j, word ) in enumerate( words ):
      for eachword in words:
        for i in range( len(words[0])):
          if word[i] == eachword[i]:
            score[j]=score[j]+1
    max_index = 0

    for i in range(1, len(score)):
      if score[i] > score[max_index]:
        max_index = i
    # return max_index
    return max_index



def get_group_word_3(words):
    score=[0] * len(words)
    splitstore=[]
    #c=freq(words) * c[j]*(entropy(split_dict))
    for ( j, word ) in enumerate( words ):
      split_dict = {}
      for (i,eachword) in enumerate(words):
        mask = reveal( eachword, word )
        if mask not in split_dict:
          split_dict[ mask ] = []
          split_dict[ mask ].append( i )
      score[j]=(len(split_dict) )*(entropy(split_dict))
      splitstore.append(split_dict)
				
    max_index = 0
   
    for i in range(1, len(score)):
      if score[i] > score[max_index]:
        max_index = i
    return max_index



def reveal( word, query ):
		# Find out the intersections between the query and the word
		mask = [ *( '_' * len( word ) ) ]
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ' '.join( mask )
def entropy(split_dict):
  entotal=0.001
  for (i, (response, split ) ) in enumerate( split_dict.items() ):
      n=len(split_dict[response])
      entotal+=n*math.log2(n) 
  return entotal
	#return model					# Return the trained model
