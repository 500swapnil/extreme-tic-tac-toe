#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import random
import signal
import time
import datetime
import copy
import hashlib
import json
x = 'x'
o = 'o'
max_score = 50000
time_start = 5
time_limit = datetime.timedelta(seconds=15.5)

z_matrix = []


def zobrist_hash(root):

	status = copy_board.board_status
	z_hash = 0
	for i in range(0, 16):
		for j in range(0, 16):
			row = root[0] + 1
			col = root[1] + 1
			if status[i][j] == o:
				z_hash = z_hash ^ z_matrix[i][j][row][col][1]
			elif status[i][j] == x:
				z_hash = z_hash ^ z_matrix[i][j][row][col][0]
	return z_hash


class Team2():
	def __init__(self):
		self.label = o
		self.turn = False
		self.max_depth = 200
		self.transposition_table = dict()
		self.start = 0
		self.i = 0
		self.initZobrist()
		self.TLD = [(1, 0), (0, 1), (1, 2), (2, 1)]
		self.TRD = [(0, 2), (1, 1), (2, 2), (1, 3)]
		self.BLD = [(2, 0), (1, 1), (2, 2), (3, 1)]
		self.BRD = [(1, 2), (2, 1), (3, 2), (2, 3)]
		self.cont_scoreBlock = [0, 10, 50, 200, 0]
		self.stop_scoreBlock = [0, 10, 50, 500, 0]
		self.cont_scoreCell = [0, 5, 20, 200, 0]
		self.stop_scoreCell = [0, 5, 20, 250, 0]
		return

	def move(self, board, prev_move, label):
		global copy_board
		copy_board = copy.deepcopy(board)
		move = copy.deepcopy(prev_move)
		self.label = label
		global order
		order = dict()
		next_guess = self.iterative_deep_search(move)
		move = None
		copy_board = None

		return next_guess

	def get_label(self, label):
		if label == o:
			return x
		else:
			return o

	def initZobrist(self):
		for i in range(16):
			z_matrix.append([])
			for j in range(16):
				z_matrix[i].append([])
				for k in range(17):
					z_matrix[i][j].append([])
					for l in range(17):
						z_matrix[i][j][k].append([])
						for m in range(2):
							z_matrix[i][j][k][l].append(
								random.getrandbits(64))

	def iterative_deep_search(self, root):
		self.start = datetime.datetime.utcnow()
		firstGuess = 0
		for l in range(1, self.max_depth):
			if datetime.datetime.utcnow() - self.start > time_limit:
				break
			self.transposition_table = dict()
			firstGuess, move = self.MTDF(root, firstGuess, l)
		return move

	def MTDF(self, root, firstGuess, depth):
		guess = firstGuess

		upperBound = max_score
		lowerBound = -max_score

		while upperBound - lowerBound > 0:
			if guess != lowerBound:
				beta = guess
			else:
				beta = guess + 1

			guess, move = self.alphaBeta(root, beta-1, beta, depth)
			self.i = 0
			self.turn = True

			if datetime.datetime.utcnow() - self.start > time_limit:
				return guess, move

			if beta < guess:
				lowerBound = guess
			else:
				upperBound = guess

		return guess, move

	def alphaBeta(self, root, alpha, beta, depth):

		board_hash = zobrist_hash(root)

		n_low = (board_hash + 1)*(board_hash + 1 + 1)/2 + 1
		n_high = (board_hash + 2)*(board_hash + 2 + 1)/2 + 2

		lower = -2*max_score, root
		upper = 2*max_score, root

		if n_low in self.transposition_table:
			lower = self.transposition_table[n_low]
			if lower[0] - beta >= 0:
				return lower

		if n_high in self.transposition_table:
			upper = self.transposition_table[n_high]
			if upper[0] - alpha <= 0:
				return upper

		status = copy_board.find_terminal_state()

		if status[1] == "WON":

			if self.turn == True:
				return -max_score, root
			else:
				return max_score, root

		alpha = max(alpha, lower[0])
		beta = min(beta, upper[0])

		hashed_move = (board_hash + self.i) * \
			(board_hash + self.i + 1)/2 + self.i
		intel_move = []

		if hashed_move not in order:
			child = copy_board.find_valid_move_cells(root)
			n_children = len(child)

		else:
			child = []
			child.extend(order[hashed_move])
			n_children = len(child)

		if depth == 0 or n_children == 0:  # base case
			answer = root
			guess = self.heuristics(root)

		elif self.turn == True:
			guess = -max_score
			answer = child[0]
			if datetime.datetime.utcnow() - self.start > time_limit:
				return guess, answer

			a = alpha
			i = 0
			while (i < n_children) & (guess < beta):

				c_matrix = child[i]
				row = c_matrix[0]/4
				col = c_matrix[1]/4
				self.i += 1
				block_value = copy_board.block_status[row][col]
				copy_board.update(root, c_matrix, self.label)

				prev_guess, prev_answer = self.alphaBeta(
					c_matrix, a, beta, depth-1)

				self.i -= 1
				copy_board.board_status[c_matrix[0]][c_matrix[1]] = '-'
				copy_board.block_status[row][col] = block_value

				prev_answer = (prev_guess, c_matrix)
				intel_move.append(prev_answer)

				if prev_guess - guess > 0:
					guess = prev_guess
					answer = c_matrix

				a = max(a, guess)
				i = i + 1

				self.turn = False
			self.turn = True

		else:
			guess = max_score
			answer = child[0]
			if datetime.datetime.utcnow() - self.start > time_limit:
				return guess, answer

			b = beta
			i = 0
			while (guess > alpha) & (i < n_children):

				c_matrix = child[i]

				row = c_matrix[0]/4
				col = c_matrix[1]/4

				block_value = copy_board.block_status[row][col]
				self.i += 1
				copy_board.update(root, c_matrix, self.get_label(self.label))


				prev_guess, prev_answer = self.alphaBeta(
					c_matrix, alpha, b, depth-1)

				copy_board.board_status[c_matrix[0]][c_matrix[1]] = '-'
				copy_board.block_status[row][col] = block_value
				self.i -= 1

				prev_answer = (prev_guess, c_matrix)
				intel_move.append(prev_answer)

				if prev_guess - guess < 0:
					answer = c_matrix
					guess = prev_guess

				b = min(b, guess)
				i += 1

				self.turn = True

			self.turn = False

		prev_answer = []

		if self.turn == False:
			intel_move = sorted(intel_move)
		else:
			intel_move = sorted(intel_move, reverse=True)

		for i in intel_move:
			prev_answer.append(i[1])
			child.remove(i[1])

		random.shuffle(child)

		order[hashed_move] = []
		order[hashed_move].extend(prev_answer)
		order[hashed_move].extend(child)

		if guess - beta >= 0:
			self.transposition_table[n_low] = guess, answer

		if guess - alpha <= 0:
			self.transposition_table[n_high] = guess, answer

		return guess, answer

	def heuristics(self, last_move):
		
		heurist = 0
		block_won = 500
		block_won_center = 125
		block_won_edge = 150
		block_won_cor = 200
		free_move = 400
		last_move_win = -500

		TLDdrawCount = 0
		TLDcountSelf = 0
		TLDcountOther = 0

		TRDdrawCount = 0
		TRDcountSelf = 0
		TRDcountOther = 0

		BRDdrawCount = 0
		BRDcountSelf = 0
		BRDcountOther = 0

		BLDdrawCount = 0
		BLDcountSelf = 0
		BLDcountOther = 0

		full = copy_board.block_status
		status = copy_board.board_status

		cont_scoreBlock = self.cont_scoreBlock
		stop_scoreBlock = self.stop_scoreBlock
		cont_scoreCell = self.cont_scoreCell
		stop_scoreCell = self.stop_scoreCell

		cX = last_move[0] % 4
		cY = last_move[1] % 4
		bX = last_move[0]/4
		bY = last_move[1]/4

		ourWin = self.label
		oppWin = self.get_label(self.label)

		if full[cX][cY] != '-':
			if not self.turn:
				heurist -= free_move
			else:
				heurist += free_move

		center_cell = 2
		cellOfCentre = 0.5
		lastBlockDraw = 15

		if full[bX][bY] == oppWin or full[bX][bY] == ourWin:
			heurist += last_move_win
			
		elif full[bX][bY] != 'd':
			if not self.turn:
				heurist += lastBlockDraw
			else:
				heurist -= lastBlockDraw

		for i in range(4):
			rowdrawCount = coldrawCount = rowcountSelf = rowcountOther = colcountSelf = colcountOther = 0
			for j in range(4):

				cellCountSelfTLD = cellCountOtherTLD = 0
				cellCountSelfTRD = cellCountOtherTRD = 0
				cellCountSelfBLD = cellCountOtherBLD = 0
				cellCountSelfBRD = cellCountOtherBRD = 0

				if full[i][j] == ourWin:
					if (i, j) in self.TLD:
						TLDcountSelf += 1

					if (i, j) in self.BLD:
						BLDcountSelf += 1

					if (i, j) in self.TRD:
						TRDcountSelf += 1

					if (i, j) in self.BRD:
						BRDcountSelf += 1

				elif full[i][j] == oppWin:
					if (i, j) in self.TLD:
						TLDcountOther += 1

					if (i, j) in self.BLD:
						BLDcountOther += 1

					if (i, j) in self.TRD:
						TRDcountOther += 1

					if (i, j) in self.BRD:
						BRDcountOther += 1

				elif full[i][j] == 'd':
					if (i, j) in self.TLD:
						TLDdrawCount += 1
					if (i, j) in self.BLD:
						BLDdrawCount += 1
					if (i, j) in self.TRD:
						TRDdrawCount += 1
					if (i, j) in self.BRD:
						BRDdrawCount += 1

				if i in [0, 3] or j in [0, 3]:
					if full[i][j] == oppWin:
						heurist -= block_won_edge
					elif full[i][j] == ourWin:
						heurist += block_won_edge
				elif i in [0, 3] and j in [0, 3]:
					if full[i][j] == oppWin:
						heurist -= block_won_cor
					elif full[i][j] == ourWin:
						heurist += block_won_cor
				elif i in [1, 2] and j in [1, 2]:
					if full[i][j] == oppWin:
						heurist -= block_won_center
					elif full[i][j] == ourWin:
						heurist += block_won_center

				if full[i][j] == oppWin:
					heurist -= block_won
				elif full[i][j] == ourWin:
					heurist += block_won

				if full[i][j] == ourWin:
					rowcountSelf += 1

				elif full[i][j] == oppWin:
					rowcountOther += 1

				elif full[i][j] != '-':
					rowdrawCount += 1

				if full[j][i] == ourWin:
					colcountSelf += 1

				elif full[j][i] == oppWin:
					colcountOther += 1

				elif full[j][i] != '-':
					coldrawCount += 1

				if full[i][j] == '-':
					for k in range(4):
						cellCountOtherRow = cellCountOtherCol = cellCountSelfRow = cellCountSelfCol = 0
						for l in range(4):
							if status[4*i+k][4*j+l] == oppWin:
								if (4*i+k, 4*j+l) in self.TLD:
									cellCountOtherTLD += 1

								if (4*i+k, 4*j+l) in self.BLD:
									cellCountOtherBLD += 1

								if (4*i+k, 4*j+l) in self.TRD:
									cellCountOtherTRD += 1

								if (4*i+k, 4*j+l) in self.BRD:
									cellCountOtherBRD += 1

								if (k == 1 or k == 2) and (l == 1 or l == 2):
									heurist -= center_cell
								elif (k == 0 or k == 3) or (l == 0 or l == 3):
									heurist -= center_cell

								if (i == 0 or i == 3) or (j == 0 or j == 3):
									heurist -= cellOfCentre
								elif (i == 1 or i == 2) and (j == 1 or j == 2):
									heurist -= cellOfCentre

							elif status[4*i+k][4*j+l] == ourWin:
								if (4*i+k, 4*j+l) in self.TLD:
									cellCountSelfTLD += 1

								if (4*i+k, 4*j+l) in self.BLD:
									cellCountSelfBLD += 1

								if (4*i+k, 4*j+l) in self.TRD:
									cellCountSelfTRD += 1

								if (4*i+k, 4*j+l) in self.BRD:
									cellCountSelfBRD += 1

								if (k == 1 or k == 2) and (l == 1 or l == 2):
									heurist += center_cell
								elif (k == 0 or k == 3) or (l == 0 or l == 3):
									heurist += center_cell
								if (i == 0 or i == 3) or (j == 0 or j == 3):
									heurist += cellOfCentre
								elif (i == 1 or i == 2) and (j == 1 or j == 2):
									heurist += cellOfCentre

							if status[4*i+k][4*j+l] == ourWin:
								cellCountSelfRow += 1
							elif status[4*i+k][4*j+l] == oppWin:
								cellCountOtherRow += 1

							if status[4*i+l][4*j+k] == ourWin:
								cellCountSelfCol += 1
							elif status[4*i+l][4*j+k] == oppWin:
								cellCountOtherCol += 1

						if cellCountSelfRow > 0:
							if cellCountOtherRow == 0:
								heurist += cont_scoreCell[cellCountSelfRow]
							else:

								heurist -= stop_scoreCell[cellCountSelfRow]
						if cellCountOtherRow > 0:
							if cellCountSelfRow == 0:
								heurist -= cont_scoreCell[cellCountOtherRow]
							else:
								heurist += stop_scoreCell[cellCountOtherRow]

						if cellCountSelfCol > 0:
							if cellCountOtherCol == 0:
								heurist += cont_scoreCell[cellCountSelfCol]
							else:

								heurist -= stop_scoreCell[cellCountSelfCol]
						if cellCountOtherCol > 0:
							if cellCountSelfCol == 0:
								heurist -= cont_scoreCell[cellCountOtherCol]
							else:
								heurist += stop_scoreCell[cellCountOtherCol]

					if cellCountSelfTLD > 0:
						if cellCountOtherTLD == 0:
							heurist += 1.2*cont_scoreCell[cellCountSelfTLD]
						else:

							heurist -= 1.2*stop_scoreCell[cellCountSelfTLD]
					if cellCountOtherTLD > 0:
						if cellCountSelfTLD == 0:
							heurist -= 1.2*cont_scoreCell[cellCountOtherTLD]
						else:
							heurist += 1.2*stop_scoreCell[cellCountOtherTLD]

					if cellCountSelfTRD > 0:
						if cellCountOtherTRD == 0:
							heurist += 1.2*cont_scoreCell[cellCountSelfTRD]
						else:

							heurist -= 1.2*stop_scoreCell[cellCountSelfTRD]
					if cellCountOtherTRD > 0:
						if cellCountSelfTRD == 0:
							heurist -= 1.2*cont_scoreCell[cellCountOtherTRD]
						else:
							heurist += 1.2*stop_scoreCell[cellCountOtherTRD]

					if cellCountSelfBLD > 0:
						if cellCountOtherBLD == 0:
							heurist += 1.2*cont_scoreCell[cellCountSelfBLD]
						else:

							heurist -= 1.2*stop_scoreCell[cellCountSelfBLD]
					if cellCountOtherBLD > 0:
						if cellCountSelfBLD == 0:
							heurist -= 1.2*cont_scoreCell[cellCountOtherBLD]
						else:
							heurist += 1.2*stop_scoreCell[cellCountOtherBLD]

					if cellCountSelfBRD > 0:
						if cellCountOtherBRD == 0:
							heurist += 1.2*cont_scoreCell[cellCountSelfBRD]
						else:

							heurist -= 1.2*stop_scoreCell[cellCountSelfBRD]
					if cellCountOtherBRD > 0:
						if cellCountSelfBRD == 0:
							heurist -= 1.2*cont_scoreCell[cellCountOtherBRD]
						else:
							heurist += 1.2*stop_scoreCell[cellCountOtherBRD]

			if rowcountSelf > 0:
				if rowcountOther + rowdrawCount > 0:
					heurist -= stop_scoreBlock[rowcountSelf]
				else:
					heurist += cont_scoreBlock[rowcountSelf]

			if rowcountOther > 0:
				if rowcountSelf + rowdrawCount > 0:
					heurist += stop_scoreBlock[rowcountOther]
				else:
					heurist -= cont_scoreBlock[rowcountOther]

			if colcountSelf > 0:
				if colcountOther + coldrawCount > 0:
					heurist -= stop_scoreBlock[colcountSelf]
				else:
					heurist += cont_scoreBlock[colcountSelf]

			if colcountOther > 0:
				if colcountSelf + coldrawCount > 0:
					heurist += stop_scoreBlock[colcountOther]
				else:
					heurist -= cont_scoreBlock[colcountOther]

		if TLDcountSelf > 0:
			if TLDcountOther + TLDdrawCount > 0:
				heurist -= 1.2*stop_scoreBlock[TLDcountSelf]
			else:
				heurist += 1.2*cont_scoreBlock[TLDcountSelf]

		if TLDcountOther > 0:
			if TLDcountSelf + TLDdrawCount > 0:
				heurist += 1.2*stop_scoreBlock[TLDcountOther]
			else:
				heurist -= 1.2*cont_scoreBlock[TLDcountOther]

		if BLDcountSelf > 0:
			if BLDcountOther + BLDdrawCount > 0:
				heurist -= 1.2*stop_scoreBlock[BLDcountSelf]
			else:
				heurist += 1.2*cont_scoreBlock[BLDcountSelf]

		if BLDcountOther > 0:
			if BLDcountSelf + BLDdrawCount > 0:
				heurist += 1.2*stop_scoreBlock[BLDcountOther]
			else:
				heurist -= 1.2*cont_scoreBlock[BLDcountOther]

		if TRDcountSelf > 0:
			if TRDcountOther + TRDdrawCount > 0:
				heurist -= 1.2*stop_scoreBlock[TRDcountSelf]
			else:
				heurist += 1.2*cont_scoreBlock[TRDcountSelf]
		if TRDcountOther > 0:
			if TRDcountSelf + TRDdrawCount > 0:
				heurist += 1.2*stop_scoreBlock[TRDcountOther]
			else:
				heurist -= 1.2*cont_scoreBlock[TRDcountOther]

		if BRDcountSelf > 0:
			if BRDcountOther + BRDdrawCount > 0:
				heurist -= 1.2*stop_scoreBlock[BRDcountSelf]
			else:
				heurist += 1.2*cont_scoreBlock[BRDcountSelf]
		if BRDcountOther > 0:
			if BRDcountSelf + BRDdrawCount > 0:
				heurist += 1.2*stop_scoreBlock[BRDcountOther]
			else:
				heurist -= 1.2*cont_scoreBlock[BRDcountOther]

		return heurist
