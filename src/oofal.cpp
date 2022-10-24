#include <iostream>
#include <vector>
#include <queue>
#include <stack>

#pragma region Graphs

template <typename E>
struct Edge {
	int startIndex;
	int endIndex;
	E value;

	bool operator<(const Edge& rhs) {
		return value < rhs.value;
	}
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

template <typename V, typename E>
std::vector<Edge<E>> Prim(IGraph<V, E>* graph) {
	auto count = graph->size();
	std::vector<Edge<E>> tree;
	tree.reserve(count - 1);

	std::vector<bool> visited(count);

	std::priority_queue<Edge<E>> pQueue;
	int vertexIndex = 0;

	for (int i = 0; i < count; i++)
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
		} while (!visited[edge.endIndex]);
		visited[edge.endIndex] = true;
		tree.push_back(edge);
		vertexIndex = edge.endIndex;
	}
}

#pragma endregion
