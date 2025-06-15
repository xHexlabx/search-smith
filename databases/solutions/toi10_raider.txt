#include <bits/stdc++.h>
using namespace std;

int m, n;
vector<pair<int,int>> even_d = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {-1, 1}};
vector<pair<int,int>> odd_d = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, -1}, {-1, -1}};

int grid[105][105];
bool visited[123][123][5000];

bool check(int i, int j) {
    return (i >= 0 && i < m && j>=0 && j<n);
}
void solve() {
    cin>> m>> n;

    for (int i = 0; i<m; i++) for (int j=0; j<n; j++) cin>> grid[i][j];
    
    int mid = (m-1)/2;
    for (int di = -1; di<=1; di++) {
        int newi = mid+di;
        if (check(newi, 0) && grid[newi][0] && grid[newi][0]==1) {
            visited[newi][0][1] = true;
        }
    }

    for (int step = 2; step<=5000; step++) {
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                if (grid[i][j]==0 || step%grid[i][j]) continue; 
                for (auto x : (i%2==0 ? even_d : odd_d)) {
                    int newi = i+x.first;
                    int newj = j+x.second;
                    if (check(newi, newj) && visited[newi][newj][step-1]) {
                        visited[i][j][step] = true;
                        break;
                    }
                }
            }
        }
        if (visited[mid][n-1][step]) {
            cout<< step;
            return;
        }
    }
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    solve();
}