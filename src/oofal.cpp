#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>

#pragma region LinearAlgebra

template <typename T>
std::vector<T> crossProduct3D(const std::vector<T>& v1, const std::vector<T>& v2) {
	std::vector<T> result(v1.size());
	result[0] = v1[1] * v2[2] - v1[2] * v2[1];
	result[1] = v1[2] * v2[0] - v1[0] * v2[2];
	result[2] = v1[0] * v2[1] - v1[1] * v2[0];
	return result;
}

template <typename T>
T dotProduct(const std::vector<T>& v1, const std::vector<T>& v2) {
	T result = 0;
	for (size_t i = 0; i < v1.size(); i++)
	{
		result += v1[i] * v2[i];
	}
	return result;
}

template <typename T>
float lengthf(const std::vector<T>& vec) {
	auto result = dotProduct(vec, vec);
	return sqrtf(result);
}

template <typename T>
double length(const std::vector<T>& vec) {
	auto result = dotProduct(vec, vec);
	return sqrt(result);
}

#pragma endregion

#pragma region Sets
/// <summary>
/// Basic idea of union-find, can be painfull to use effectively 
/// TODO: Improve this in the future
/// </summary>
struct UnionFindSet {
	struct Node {
		int parent;
		int rank;
	};

	std::vector<Node> set;

	int find(int index) {
		if (set[index].parent == index) {
			return index;
		}
		set[index].parent = find(set[index].parent);
		return set[index].parent;
	}

	void union_(int index1, int index2) {
		if (index1 >= set.size()) set.push_back({ index1, 0 });
		if (index2 >= set.size()) set.push_back({ index1, 0 });

		int set1 = find(index1);
		int set2 = find(index2);

		if (set1 == set2) return;

		if (set[set1].rank >= set[set2].rank) {
			set[set1].rank++;

			set[set2].parent = set1;
			set[set2].rank = 0;
		}
		else {
			set[set2].rank++;

			set[set1].parent = set2;
			set[set1].rank = 0;
		}
	}
};

/// <summary>
/// Use this if you need to use custom data as your set's key
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
struct UnionFindMap {
	struct Node {
		int index;
		int parent;
		int rank;
	};

	std::unordered_map<T, Node> map;
	std::vector<Node> set;

	int find(int index) {
		if (set[index].parent == index) {
			return index;
		}
		set[index].parent = find(set[index].parent);
		return set[index].parent;
	}

	int findByValue(T val) {
		if (map.count(val) <= 0) return -1;

		return find(map[val].index);
	}

	void union_(int index1, int index2) {
		int set1 = find(index1);
		int set2 = find(index2);

		if (set1 == set2) return;

		if (set[set1].rank >= set[set2].rank) {
			set[set1].rank++;

			set[set2].parent = set1;
			set[set2].rank = 0;
		}
		else {
			set[set2].rank++;

			set[set1].parent = set2;
			set[set1].rank = 0;
		}
	}

	void unionByValue(T val1, T val2) {
		auto val1Exists = map.count(val1) > 0;
		auto indexVal1 = -1;
		if (!val1Exists) {
			indexVal1 = set.size();
			Node node = { indexVal1, indexVal1, 0 };
			set.push_back(node);
			map[val1] = node;
		}
		else {
			indexVal1 = map[val1].index;
		}

		auto val2Exists = map.count(val2) > 0;
		auto indexVal2 = -1;
		if (!val2Exists) {
			indexVal2 = set.size();
			Node node = { indexVal2, indexVal2, 0 };
			set.push_back(node);
			map[val2] = node;
		}
		else {
			indexVal2 = map[val2].index;
		}

		union_(indexVal1, indexVal2);
	}
};

#pragma endregion



#pragma region Graphs

template <typename E>
struct Edge {
	int startIndex;
	int endIndex;
	E value;
};

template <typename V, typename E>
struct Vertex {
	V value;
	std::vector<Edge<E>> edges;
};

template <typename V, typename E>
class IGraph {
public:
	virtual int size() const = 0;

	virtual int addVertex(V vertex) = 0;

	virtual void setVertex(int index, V value) = 0;

	virtual V getVertex(int index) = 0;

	virtual void addEdge(int v1, int v2, E value) = 0;

	virtual void addEdgeBothWays(int v1, int v2, E value) = 0;

	virtual E getEdge(int v1, int v2) = 0;

	virtual std::vector<int> getNeighboursIndices(int vertexIndex) = 0;

	virtual Vertex<V, E> getVertexContainer(int vertexIndex) = 0;

	virtual std::vector<Edge<E>> getNeighboursEdges(int vertexIndex) = 0;
};


template <typename V, typename E>
class AdjentencyGraph : public IGraph<V, E> {
private:
	std::vector<Vertex<V, E>> vertices;

public:

	AdjentencyGraph() {

	}

	AdjentencyGraph(int verticesCount, V defaultValue) : vertices(verticesCount, { defaultValue })
	{

	}

	AdjentencyGraph(IGraph<V, E>* other) : vertices(other->size()) {
		auto count = other->size();
		for (int i = 0; i < count; i++)
		{
			this->setVertex(i, other->getVertex(i));
			auto edges = other->getNeighboursIndices(i);
			for (auto edge : edges) {
				// can be faster by adding a function returning an edge with it's weight
				// will be added in the future
				this->addEdge(i, edge, other->getEdge(i, edge));
			}
		}
	}

	inline int size() const override {
		return vertices.size();
	}

	inline int addVertex(V vertex) override {
		this->vertices.push_back({ vertex });
		return this->vertices.size() - 1;
	}

	inline void setVertex(int index, V value) override {
		this->vertices[index].value = value;
	}

	inline V getVertex(int index) override {
		return this->vertices[index].value;
	}

	inline void addEdge(int v1, int v2, E value) override {
		this->vertices[v1].edges.push_back({ v1, v2, value });
	}

	inline void addEdgeBothWays(int v1, int v2, E value) override {
		this->addEdge(v1, v2, value);
		this->addEdge(v2, v1, value);
	}

	inline E getEdge(int v1, int v2) override {
		for (const auto& edge : this->vertices[v1].edges) {
			if (edge.endIndex == v2) {
				return edge.value;
			}
		}
		return 0;
	}

	inline std::vector<int> getNeighboursIndices(int vertexIndex) override {
		std::vector<int> output;
		output.reserve(this->vertices[vertexIndex].edges.size());

		for (const auto& edge : this->vertices[vertexIndex].edges) {
			output.push_back(edge.endIndex);
		}

		return output;
	}

	inline Vertex<V, E> getVertexContainer(int vertexIndex) override {
		return this->vertices[vertexIndex];
	}

	inline std::vector<Edge<E>> getNeighboursEdges(int vertexIndex) override {
		return this->vertices[vertexIndex].edges;
	}
};

template <typename V, typename E>
class MatrixGraph : public IGraph<V, E> {
private:
	int capacity;
	std::vector<V> vertices;
	std::vector<E> edges;

	inline int map2Dto1D(int x, int y) {
		return x * this->capacity + y;
	}

public:
	inline int size() const override {
		return vertices.size();
	}

	MatrixGraph(int capacity) : capacity(capacity), edges(capacity* capacity)
	{
		this->vertices.reserve(capacity);
	}

	MatrixGraph(int capacity, V value) : capacity(capacity), edges(capacity* capacity), vertices(capacity, value)
	{

	}

	MatrixGraph(IGraph<V, E>* other) :capacity(other->size()), vertices(other->size()), edges(other->size()* other->size()) {
		auto count = other->size();
		for (int i = 0; i < count; i++)
		{
			this->setVertex(i, other->getVertex(i));
			auto edges = other->getNeighboursIndices(i);
			for (auto edge : edges) {
				// can be faster by adding a function returning an edge with it's weight
				// will be added in the future
				this->addEdge(i, edge, other->getEdge(i, edge));
			}
		}
	}


	inline int addVertex(V vertex) override {
		if (this->vertices.size() == this->capacity) throw std::out_of_range("Graph is full");
		this->vertices.push_back(vertex);
		return this->vertices.size() - 1;
	}

	inline void setVertex(int index, V value) override {
		this->vertices[index] = value;
	}

	inline V getVertex(int index) override {
		return this->vertices[index];
	}

	inline void addEdge(int v1, int v2, E value) override {
		this->edges[this->map2Dto1D(v1, v2)] = value;
	}

	inline void addEdgeBothWays(int v1, int v2, E value) override {
		this->addEdge(v1, v2, value);
		this->addEdge(v2, v1, value);
	}

	inline E getEdge(int v1, int v2) override {
		return this->edges[this->map2Dto1D(v1, v2)];
	}

	inline std::vector<int> getNeighboursIndices(int vertexIndex) override {
		std::vector<int> output;
		for (int i = 0; i < this->capacity; i++)
		{
			if (this->getEdge(vertexIndex, i) != 0) {
				output.push_back(i);
			}
		}
		return output;
	}

	inline Vertex<V, E> getVertexContainer(int vertexIndex) override {
		return this->vertices[vertexIndex];
	}

	inline std::vector<Edge<E>> getNeighboursEdges(int vertexIndex) override {
		std::vector<Edge<E>> output;
		for (int i = 0; i < this->capacity; i++)
		{
			auto edgeValue = this->getEdge(vertexIndex, i);
			if (edgeValue != 0) {
				output.push_back({ vertexIndex, i, edgeValue });
			}
		}
		return output;
	}
};

#pragma endregion

#pragma region GraphSearching

template <typename V, typename E>
bool breadthFirstSearch(IGraph<V, E>* graph, std::vector<bool>& visited, int rootIndex, int goal = -1) {
	std::queue<int> queue;

	queue.push(rootIndex);
	visited[rootIndex] = true;

	while (!queue.empty()) {
		auto index = queue.front();
		queue.pop();
		// std::cout << index << " ";
		if (goal != -1 && index == goal) {
			return true;
		}
		auto neighbours = graph->getNeighboursIndices(index);
		for (const auto& i : neighbours) {
			if (!visited[i]) {
				queue.push(i);
				visited[i] = true;
			}
		}
	}

	return false;
}

template <typename V, typename E>
bool depthFirstSearch(IGraph<V, E>* graph, std::vector<bool>& visited, int rootIndex, int goal = -1) {
	std::stack<int> stack;

	stack.push(rootIndex);
	visited[rootIndex] = true;

	while (!stack.empty()) {
		auto index = stack.top();
		stack.pop();
		// std::cout << index << " ";
		if (goal != -1 && index == goal) {
			return true;
		}
		auto neighbours = graph->getNeighboursIndices(index);
		for (const auto& i : neighbours) {
			if (!visited[i]) {
				stack.push(i);
				visited[i] = true;
			}
		}
	}

	return false;
}

#pragma endregion

#pragma region GraphCharacteristics

template <typename V, typename E>
int getComponentsCount(IGraph<V, E>* graph) {
	auto count = graph->size();
	int components = 0;

	std::vector<bool> explored(count, false);
	std::stack<int> stack;

	for (int i = 0; i < count; i++)
	{
		if (!explored[i]) {
			components++;
			depthFirstSearch(graph, explored, i);
		}
	}

	return components;
}

template <typename V, typename E>
bool isBipartite(IGraph<V, E>* graph, std::vector<int>& visited, int rootIndex) {
	std::stack<int> stack;

	stack.push(rootIndex);
	visited[rootIndex] = 1;

	while (!stack.empty()) {
		auto index = stack.top();
		stack.pop();
		auto neighbours = graph->getNeighboursIndices(index);
		for (const auto& i : neighbours) {
			if (visited[i] == 0) {
				stack.push(i);
				visited[i] = -visited[index];
			}
			if (visited[i] == visited[index]) {
				return false;
			}
		}
	}
	return true;
}

template <typename V, typename E>
bool isBipartite(IGraph<V, E>* graph) {
	auto count = graph->size();

	std::vector<int> explored(count);

	for (int i = 0; i < count; i++)
	{
		if (!explored[i]) {
			if (!isBipartite(graph, explored, i)) {
				return false;
			}
		}
	}

	return true;
}

template <typename V, typename E>
bool hasCycles(IGraph<V, E>* graph, std::vector<bool>& visited, int rootIndex) {
	std::stack<int> stack;

	stack.push(rootIndex);
	stack.push(-1);

	visited[rootIndex] = true;

	while (!stack.empty()) {
		auto previousIndex = stack.top();
		stack.pop();
		auto index = stack.top();
		stack.pop();

		auto neighbours = graph->getNeighboursIndices(index);
		for (const auto& i : neighbours) {
			if (!visited[i]) {
				stack.push(i);
				stack.push(index);
				visited[i] = true;
			}
			else if (i != previousIndex) {
				return true;
			}
		}
	}
	return false;
}

template <typename V, typename E>
bool hasCycles(IGraph<V, E>* graph) {
	auto count = graph->size();

	std::vector<bool> explored(count);

	for (int i = 0; i < count; i++)
	{
		if (!explored[i]) {
			if (hasCycles(graph, explored, i)) {
				return true;
			}
		}
	}

	return false;
}

struct GraphCharacteristics {
	int components;
	bool cycles;
	bool bipartite;
	bool tree;
};

template <typename V, typename E>
GraphCharacteristics computeCharacteristics(IGraph<V, E>* graph) {
	auto count = graph->size();

	std::vector<int> visited(count);
	std::stack<int> stack;

	int components = 0;
	bool cycles = false;
	bool bipartite = true;

	for (int v = 0; v < count; v++)
	{
		if (visited[v] != 0)
		{
			continue;
		}

		stack.push(v);
		stack.push(-1);
		visited[v] = 1;
		components++;

		while (!stack.empty()) {
			auto previousIndex = stack.top();
			stack.pop();
			auto index = stack.top();
			stack.pop();
			auto neighbours = graph->getNeighboursIndices(index);
			for (const auto& i : neighbours) {
				if (visited[i] == 0) {
					stack.push(i);
					stack.push(index);
					visited[i] = -visited[index];
				}
				else {
					if (visited[i] == visited[index]) {
						bipartite = false;
					}
					if (i != previousIndex) {
						cycles = true;
					}
				}
			}
		}
	}

	return { components, cycles, bipartite, (!cycles && components == 1) };
}

#pragma endregion

#pragma region MinimalSpanningTrees

template <typename E>
struct MinimalSpanningTree {
	std::vector<Edge<E>> edges;
	E edgesSum;
	bool graphDisconnected;
};

template <typename V, typename E>
MinimalSpanningTree<E> MST_Prim(IGraph<V, E>* graph) {
	auto count = graph->size();
	E sum = 0;
	std::vector<Edge<E>> tree;
	tree.reserve(count - 1);

	std::vector<bool> visited(count, false);
	auto compare = [](Edge<E> const& a, Edge<E> const& b) { return a.value > b.value; };
	std::priority_queue<Edge<E>, std::vector<Edge<E>>, decltype(compare)> pQueue(compare);
	int vertexIndex = 0;
	visited[vertexIndex] = true;

	for (int i = 0; i < count - 1; i++)
	{
		auto edges = graph->getNeighboursEdges(vertexIndex);
		for (const auto& edge : edges) {
			if (!visited[edge.endIndex]) {
				pQueue.push(edge);
			}
		}
		auto edge = pQueue.top();
		do {
			edge = pQueue.top();
			pQueue.pop();
		} while (visited[edge.endIndex] && !pQueue.empty());

		if (visited[edge.endIndex]) {
			return { std::vector<Edge<E>>(), 0, true};
		}

		visited[edge.endIndex] = true;
		tree.push_back(edge);
		sum += edge.value;
		vertexIndex = edge.endIndex;
	}
	return { tree, sum, false };
}

#pragma endregion


template <typename E>
struct by_v1 {
	bool operator()(Edge<E> const& a, Edge<E> const& b) const {
		return a.startIndex < b.startIndex;
	}
};


void UnionFindSolution() {
	char operation;
	std::string IP1_string = "", IP2_string = "";
	UnionFindMap<long long> unionFind;
	while (std::cin >> operation) {
		std::cin >> IP1_string >> IP2_string;

		IP1_string.erase(remove(IP1_string.begin(), IP1_string.end(), '.'), IP1_string.end());
		IP2_string.erase(remove(IP2_string.begin(), IP2_string.end(), '.'), IP2_string.end());
		long long IP1 = stoll(IP1_string);
		long long IP2 = stoll(IP2_string);
		if (operation == 'T') {
			auto set1 = unionFind.findByValue(IP1);
			auto set2 = unionFind.findByValue(IP2);
			std::cout << (set1 != -1 && set2 != -1 && set1 == set2 ? "T" : "N") << "\n";
		}
		else {
			unionFind.unionByValue(IP1, IP2);
		}
	}
}

int main()
{
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(NULL);
	// std::cout.tie(NULL);

	UnionFindSolution();
	/*int tests = 0;
	std::cin >> tests;
	while (tests--) {
		int vertices = 0, edges = 0;
		std::cin >> vertices >> edges;
		auto graph = new AdjentencyGraph<int, int>(vertices, 0);

		for (int i = 0; i < edges; i++)
		{
			int v1, v2, value;
			std::cin >> v1 >> v2 >> value;
			graph->addEdgeBothWays(v1-1, v2-1, value);
		}


		auto tree = MST_Prim(graph);
		if (tree.graphDisconnected) std::cout << "brak\n";
		else std::cout << tree.edgesSum << "\n";

		delete graph;
	}*/
	// _getch();
}