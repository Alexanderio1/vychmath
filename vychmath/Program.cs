using System;
using System.Globalization;

namespace NumericalMethods
{
    /// <summary>
    /// Класс для решения СЛАУ методом Гаусса.
    /// - usePivot = false: без выбора главного элемента,
    /// - usePivot = true: полный выбор главного элемента (по всей подматрице).
    /// </summary>
    public static class GaussSolver
    {
        /// <summary>
        /// Решает систему, заданную расширенной матрицей [A|b].
        /// </summary>
        /// <param name="augmentedMatrix">Расширенная матрица размера n x (n+1)</param>
        /// <param name="usePivot">
        /// Если false – без выбора главного элемента,
        /// Если true – полный выбор главного элемента (по всей подматрице).
        /// </param>
        /// <returns>Вектор решения (длины n).</returns>
        public static float[] Solve(float[,] augmentedMatrix, bool usePivot = false)
        {
            int n = augmentedMatrix.GetLength(0);
            int m = augmentedMatrix.GetLength(1); // должно быть n+1

            // Копируем матрицу, чтобы не портить исходную
            float[,] matrix = (float[,])augmentedMatrix.Clone();

            if (!usePivot)
            {
                // Вариант без выбора главного элемента
                SolveWithoutPivot(matrix, n, m);
                // Обычный обратный ход
                return BackSubstitution(matrix, n, m);
            }
            else
            {
                // Вариант полного выбора главного элемента
                return SolveWithFullPivot(matrix, n, m);
            }
        }

        /// <summary>
        /// Прямой ход без выбора главного элемента.
        /// </summary>
        private static void SolveWithoutPivot(float[,] matrix, int n, int m)
        {
            for (int i = 0; i < n; i++)
            {
                float pivot = matrix[i, i];
                if (Math.Abs(pivot) < 1e-6)
                    throw new Exception($"Нулевой или слишком маленький ведущий элемент в строке {i + 1}.");

                // Исключаем переменную x_i из всех строк ниже
                for (int j = i + 1; j < n; j++)
                {
                    float factor = matrix[j, i] / pivot;
                    for (int k = i; k < m; k++)
                    {
                        matrix[j, k] -= factor * matrix[i, k];
                    }
                }
            }
        }

        /// <summary>
        /// Полный выбор главного элемента (complete pivoting) по всей оставшейся подматрице.
        /// </summary>
        private static float[] SolveWithFullPivot(float[,] mat, int n, int m)
        {
            // Массив перестановок столбцов: columnPerm[j] = j изначально.
            // Если мы поменяем местами столбцы i и p, то переставим и columnPerm.
            // Это нужно, чтобы в конце вернуть переменные на исходные места.
            int[] columnPerm = new int[n];
            for (int j = 0; j < n; j++)
                columnPerm[j] = j;

            // Прямой ход
            for (int i = 0; i < n; i++)
            {
                // Ищем максимальный по модулю элемент в подматрице (строки i..n-1, столбцы i..n-1)
                int pivotRow = i;
                int pivotCol = i;
                float maxVal = Math.Abs(mat[i, i]);

                for (int r = i; r < n; r++)
                {
                    for (int c = i; c < n; c++)
                    {
                        float val = Math.Abs(mat[r, c]);
                        if (val > maxVal)
                        {
                            maxVal = val;
                            pivotRow = r;
                            pivotCol = c;
                        }
                    }
                }

                // Меняем местами строки (i, pivotRow), если нужно
                if (pivotRow != i)
                {
                    for (int k = 0; k < m; k++)
                    {
                        float temp = mat[i, k];
                        mat[i, k] = mat[pivotRow, k];
                        mat[pivotRow, k] = temp;
                    }
                }

                // Меняем местами столбцы (i, pivotCol), если нужно
                // При этом нужно переставить соответствующие элементы в columnPerm.
                if (pivotCol != i)
                {
                    for (int k = 0; k < n; k++)
                    {
                        float temp = mat[k, i];
                        mat[k, i] = mat[k, pivotCol];
                        mat[k, pivotCol] = temp;
                    }
                    // Переставляем запись о переменных
                    int tmpIndex = columnPerm[i];
                    columnPerm[i] = columnPerm[pivotCol];
                    columnPerm[pivotCol] = tmpIndex;
                }

                // Проверяем ведущий элемент
                float pivot = mat[i, i];
                if (Math.Abs(pivot) < 1e-6)
                    throw new Exception($"Нулевой или слишком маленький ведущий элемент на шаге {i + 1} (после полного выбора).");

                // Гауссово исключение для строк ниже i
                for (int r = i + 1; r < n; r++)
                {
                    float factor = mat[r, i] / pivot;
                    for (int c = i; c < m; c++)
                    {
                        mat[r, c] -= factor * mat[i, c];
                    }
                }
            }

            // Обратный ход в "новом" порядке
            float[] xTemp = new float[n];
            for (int i = n - 1; i >= 0; i--)
            {
                float sum = 0;
                for (int j = i + 1; j < n; j++)
                    sum += mat[i, j] * xTemp[j];
                xTemp[i] = (mat[i, n] - sum) / mat[i, i];
            }

            // xTemp[i] соответствует переменной, которая теперь стоит на позиции i.
            // Нужно вернуть x[] так, чтобы x[columnPerm[i]] = xTemp[i].
            float[] x = new float[n];
            for (int i = 0; i < n; i++)
            {
                int originalIndex = columnPerm[i];
                x[originalIndex] = xTemp[i];
            }
            return x;
        }

        /// <summary>
        /// Стандартный обратный ход для случая без перестановок столбцов.
        /// </summary>
        private static float[] BackSubstitution(float[,] matrix, int n, int m)
        {
            float[] x = new float[n];
            for (int i = n - 1; i >= 0; i--)
            {
                float sum = 0;
                for (int j = i + 1; j < n; j++)
                {
                    sum += matrix[i, j] * x[j];
                }
                x[i] = (matrix[i, m - 1] - sum) / matrix[i, i];
            }
            return x;
        }
    }

    /// <summary>
    /// Класс для решения системы методом простых итераций (x = alpha*x + beta).
    /// </summary>
    public static class IterativeSolver
    {
        public static float[] SolveIterative(float[,] A, float[] b, float epsilon, int maxIterations, bool checkConvergence = false)
        {
            int n = A.GetLength(0);
            float[,] alpha = new float[n, n];
            float[] beta = new float[n];

            // Формируем alpha, beta
            for (int i = 0; i < n; i++)
            {
                if (Math.Abs(A[i, i]) < 1e-6)
                    throw new Exception($"Диагональный элемент A[{i + 1},{i + 1}] равен нулю или слишком мал.");
                beta[i] = b[i] / A[i, i];
                for (int j = 0; j < n; j++)
                {
                    alpha[i, j] = (i == j) ? 0 : -A[i, j] / A[i, i];
                }
            }

            // Опциональная проверка ||alpha||∞ < 1
            if (checkConvergence)
            {
                float normAlpha = 0;
                for (int i = 0; i < n; i++)
                {
                    float rowSum = 0;
                    for (int j = 0; j < n; j++)
                        rowSum += Math.Abs(alpha[i, j]);
                    normAlpha = Math.Max(normAlpha, rowSum);
                }
                if (normAlpha >= 1)
                    Console.WriteLine($"Предупреждение: условие сходимости не выполнено (||alpha|| = {normAlpha} >= 1).");
                else
                    Console.WriteLine($"Условие сходимости выполнено: ||alpha|| = {normAlpha}.");
            }

            // Итерационный процесс
            float[] xOld = new float[n];
            float[] xNew = new float[n];
            Array.Copy(beta, xOld, n);

            int iterations = 0;
            while (iterations < maxIterations)
            {
                // xNew = alpha*xOld + beta
                for (int i = 0; i < n; i++)
                {
                    float sum = 0;
                    for (int j = 0; j < n; j++)
                        sum += alpha[i, j] * xOld[j];
                    xNew[i] = beta[i] + sum;
                }

                // Проверяем разность
                float diff = 0;
                for (int i = 0; i < n; i++)
                {
                    diff = Math.Max(diff, Math.Abs(xNew[i] - xOld[i]));
                }

                if (diff < epsilon)
                    break;

                Array.Copy(xNew, xOld, n);
                iterations++;
            }

            if (iterations >= maxIterations)
                Console.WriteLine($"Достигнуто максимальное число итераций ({maxIterations}).");
            else
                Console.WriteLine($"Итерационный процесс завершился за {iterations} итераций.");

            return xNew;
        }
    }

    /// <summary>
    /// Класс для решения трёхдиагональной системы методом прогонки (Томаса).
    /// </summary>
    public static class TridiagonalSolver
    {
        public static float[] Solve(float[] c, float[] d, float[] e, float[] b)
        {
            int n = d.Length;
            if (c.Length != n || e.Length != n || b.Length != n)
                throw new ArgumentException("Размеры массивов c, d, e, b должны совпадать.");

            float[] alpha = new float[n];
            float[] beta = new float[n];

            if (Math.Abs(d[0]) < 1e-6f)
                throw new Exception("Нулевой или слишком маленький элемент d[0] на главной диагонали.");

            alpha[0] = e[0] / d[0];
            beta[0] = b[0] / d[0];

            for (int i = 1; i < n - 1; i++)
            {
                float denom = d[i] - c[i] * alpha[i - 1];
                if (Math.Abs(denom) < 1e-6f)
                    throw new Exception($"Нулевой или слишком маленький знаменатель на шаге {i}.");

                alpha[i] = e[i] / denom;
                beta[i] = (b[i] - c[i] * beta[i - 1]) / denom;
            }

            int iLast = n - 1;
            float denomLast = d[iLast] - c[iLast] * alpha[iLast - 1];
            if (Math.Abs(denomLast) < 1e-6f)
                throw new Exception($"Нулевой или слишком маленький знаменатель на шаге {iLast}.");

            beta[iLast] = (b[iLast] - c[iLast] * beta[iLast - 1]) / denomLast;

            float[] x = new float[n];
            x[iLast] = beta[iLast];
            for (int i = n - 2; i >= 0; i--)
                x[i] = beta[i] - alpha[i] * x[i + 1];

            return x;
        }
    }
}

namespace UnifiedSolver
{
    using NumericalMethods;

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Универсальное решение систем уравнений");
            Console.WriteLine("--------------------------------------");
            Console.WriteLine("Выберите метод решения:");
            Console.WriteLine("1 - Метод Гаусса (без выбора / полный выбор главного элемента)");
            Console.WriteLine("2 - Итерационный метод (простые итерации)");
            Console.WriteLine("3 - Метод прогонки для трёхдиагональной системы");
            Console.Write("Ваш выбор (1/2/3): ");
            string choice = Console.ReadLine();

            switch (choice)
            {
                case "1":
                    SolveGauss();
                    break;
                case "2":
                    SolveIterative();
                    break;
                case "3":
                    SolveTridiagonal();
                    break;
                default:
                    Console.WriteLine("Неверный выбор.");
                    break;
            }

            Console.WriteLine("\nНажмите любую клавишу для выхода...");
            Console.ReadKey();
        }

        // Решение СЛАУ методом Гаусса
        static void SolveGauss()
        {
            try
            {
                Console.WriteLine("\nМетод Гаусса");
                Console.Write("Введите количество уравнений (n): ");
                int n = int.Parse(Console.ReadLine());
                float[,] augmentedMatrix = new float[n, n + 1];

                Console.WriteLine($"Введите расширенную матрицу [A|b] (каждая строка содержит {n + 1} чисел):");
                for (int i = 0; i < n; i++)
                {
                    Console.Write($"Строка {i + 1}: ");
                    string[] tokens = Console.ReadLine()
                                             .Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (tokens.Length != n + 1)
                    {
                        Console.WriteLine("Неверное количество чисел. Попробуйте снова.");
                        i--;
                        continue;
                    }
                    for (int j = 0; j < n + 1; j++)
                        augmentedMatrix[i, j] = float.Parse(tokens[j], CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВыберите метод решения:");
                Console.WriteLine("1 - Без выбора главного элемента");
                Console.WriteLine("2 - Полный выбор главного элемента (по всей подматрице)");
                Console.Write("Ваш выбор (1/2): ");
                int methodChoice = int.Parse(Console.ReadLine());
                bool usePivot = (methodChoice == 2);

                float[] solution = GaussSolver.Solve(augmentedMatrix, usePivot);
                Console.WriteLine("\nРешение системы методом Гаусса:");
                for (int i = 0; i < solution.Length; i++)
                    Console.WriteLine($"x[{i + 1}] = {solution[i]}");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Ошибка: " + ex.Message);
            }
        }

        // Решение СЛАУ итерационным методом
        static void SolveIterative()
        {
            try
            {
                Console.WriteLine("\nИтерационный метод (метод простых итераций)");
                Console.Write("Введите количество уравнений (n): ");
                int n = int.Parse(Console.ReadLine());

                float[,] A = new float[n, n];
                float[] b = new float[n];

                Console.WriteLine("\nВведите матрицу A (n строк, по n чисел в каждой):");
                for (int i = 0; i < n; i++)
                {
                    Console.Write($"Строка {i + 1}: ");
                    string[] tokens = Console.ReadLine()
                                             .Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (tokens.Length != n)
                    {
                        Console.WriteLine("Неверное количество чисел. Попробуйте снова.");
                        i--;
                        continue;
                    }
                    for (int j = 0; j < n; j++)
                        A[i, j] = float.Parse(tokens[j], CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВведите вектор свободных членов b (n чисел):");
                {
                    string[] tokens = Console.ReadLine()
                                             .Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (tokens.Length != n)
                        throw new Exception("Неверное количество чисел во векторе b.");
                    for (int i = 0; i < n; i++)
                        b[i] = float.Parse(tokens[i], CultureInfo.InvariantCulture);
                }

                Console.Write("\nВведите требуемую точность (например, 0.001): ");
                float epsilon = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);

                Console.Write("Введите максимальное число итераций: ");
                int maxIter = int.Parse(Console.ReadLine());

                Console.Write("Проверять условие сходимости (||alpha|| < 1)? (1 - Да, 0 - Нет): ");
                bool checkConv = (Console.ReadLine() == "1");

                float[] sol = IterativeSolver.SolveIterative(A, b, epsilon, maxIter, checkConv);

                Console.WriteLine("\nРешение системы методом итераций:");
                for (int i = 0; i < sol.Length; i++)
                    Console.WriteLine($"x[{i + 1}] = {sol[i]}");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Ошибка: " + ex.Message);
            }
        }

        // Решение трёхдиагональной системы методом прогонки
        static void SolveTridiagonal()
        {
            try
            {
                Console.WriteLine("\nМетод прогонки для трёхдиагональной системы");
                Console.Write("Введите размерность системы (n): ");
                int n = int.Parse(Console.ReadLine());

                float[] c = new float[n];
                float[] d = new float[n];
                float[] e = new float[n];
                float[] b = new float[n];

                Console.WriteLine("\nВведите главную диагональ d[i] (i=0..n-1):");
                for (int i = 0; i < n; i++)
                {
                    Console.Write($"d[{i}] = ");
                    d[i] = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВведите поддиагональ c[i] (i=1..n-1, c[0] можно 0):");
                for (int i = 1; i < n; i++)
                {
                    Console.Write($"c[{i}] = ");
                    c[i] = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВведите наддиагональ e[i] (i=0..n-2, e[n-1] можно 0):");
                for (int i = 0; i < n - 1; i++)
                {
                    Console.Write($"e[{i}] = ");
                    e[i] = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВведите вектор правых частей b[i] (i=0..n-1):");
                for (int i = 0; i < n; i++)
                {
                    Console.Write($"b[{i}] = ");
                    b[i] = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }

                float[] solution = TridiagonalSolver.Solve(c, d, e, b);

                Console.WriteLine("\nРешение трёхдиагональной системы:");
                for (int i = 0; i < n; i++)
                {
                    Console.WriteLine($"x[{i}] = {solution[i]}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Ошибка: " + ex.Message);
            }
        }
    }
}
