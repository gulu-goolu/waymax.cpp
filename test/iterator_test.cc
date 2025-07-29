#include <iostream>

struct Node {
  int val = 0;

  Node *parent = nullptr, *left = nullptr, *right = nullptr;

  Node(int val, Node* parent = nullptr) : val(val), parent(parent) {}
};

struct MyIterator {
  MyIterator(Node* _ptr) : ptr(_ptr) {
    // initialize
    while (ptr->left) {
      ptr = ptr->left;
    }
  }

  MyIterator& operator++() {
    next();

    return *this;
  }

  MyIterator& operator--() {
    prev();
    return *this;
  }

  int operator*() { return ptr->val; }

  void next() {
    if (ptr->right) {
      ptr = ptr->right;
      while (ptr->left) {
        ptr = ptr->left;
      }
    } else if (ptr->parent && ptr->parent->left == ptr) {
      ptr = ptr->parent;
    } else {
      while (ptr->parent && ptr->parent->right == ptr) {
        ptr = ptr->parent;
      }
      ptr = ptr->parent;
    }
  }

  void prev() {
    if (ptr->left) {
      ptr = ptr->left;
      while (ptr->right) {
        ptr = ptr->right;
      }
    } else if (ptr->parent && ptr->parent->right == ptr) {
      ptr = ptr->parent;
    } else {
      while (ptr->parent && ptr->parent->left == ptr) {
        ptr = ptr->parent;
      }
      ptr = ptr->parent;
    }
  }

  Node* ptr;
};

Node* build1() {
  // 1. 2. 3
  auto root = new Node(2);
  root->left = new Node(1, root);
  root->right = new Node(3, root);
  return root;
}

Node* build2() {
  // 1, 2, 3, 4, 5, 6, 7
  auto root = new Node(4);
  root->left = new Node(2, root);
  root->left->left = new Node(1, root->left);
  root->left->right = new Node(3, root->left);

  root->right = new Node(6, root);
  root->right->left = new Node(5, root->right);
  root->right->right = new Node(7, root->right);
  return root;
}

int main(int argc, char* argv[]) {
  auto iter = MyIterator(build1());
  for (int i = 0; i < 3; ++i, ++iter) {
    std::cout << "iter1: " << *iter << std::endl;
  }

  auto iter2 = MyIterator(build2());
  for (int i = 0; i < 6; ++i, ++iter2) {
    std::cout << "iter2: " << *iter2 << std::endl;
  }
  for (int i = 0; i < 7; ++i, --iter2) {
    std::cout << "inv inter2: " << *iter2 << std::endl;
  }

  return 0;
}
