using System;
using System.Globalization;

namespace NumericalMethods
{
    /// <summary>
    /// Класс для решения СЛАУ методом Гаусса.
    /// Можно решать как с выбором главного элемента, так и без него.
    /// </summary>
    public static class GaussSolver
    {
        /// <summary>
        /// Решает систему, заданную расширенной матрицей [A|b].
        /// </summary>
        /// <param name="augmentedMatrix">Расширенная матрица размера n x (n+1)</param>
        /// <param name="usePivot">Если true – применяется выбор главного элемента</param>
        /// <returns>Вектор решения</returns>
        public static float[] Solve(float[,] augmentedMatrix, bool usePivot = false)
        {
            int n = augmentedMatrix.GetLength(0);
            int m = augmentedMatrix.GetLength(1);
            // Клонируем матрицу, чтобы не изменять исходную
            float[,] matrix = (float[,])augmentedMatrix.Clone();

            if (usePivot)
                SolveWithPivot(matrix, n, m);
            else
                SolveWithoutPivot(matrix, n, m);

            return BackSubstitution(matrix, n, m);
        }

        private static void SolveWithoutPivot(float[,] matrix, int n, int m)
        {
            for (int i = 0; i < n; i++)
            {
                float pivot = matrix[i, i];
                if (Math.Abs(pivot) < 1e-6)
                    throw new Exception($"Нулевой или слишком маленький ведущий элемент в строке {i + 1}.");
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

        private static void SolveWithPivot(float[,] matrix, int n, int m)
        {
            for (int i = 0; i < n; i++)
            {
                // Выбор строки с максимальным по модулю элементом в столбце i
                int maxRow = i;
                for (int j = i + 1; j < n; j++)
                {
                    if (Math.Abs(matrix[j, i]) > Math.Abs(matrix[maxRow, i]))
                        maxRow = j;
                }
                if (maxRow != i)
                    SwapRows(matrix, i, maxRow, m);

                float pivot = matrix[i, i];
                if (Math.Abs(pivot) < 1e-6)
                    throw new Exception($"Нулевой или слишком маленький ведущий элемент после выбора в строке {i + 1}.");

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

        private static void SwapRows(float[,] matrix, int row1, int row2, int m)
        {
            for (int k = 0; k < m; k++)
            {
                float temp = matrix[row1, k];
                matrix[row1, k] = matrix[row2, k];
                matrix[row2, k] = temp;
            }
        }

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
    /// Класс для решения системы методом простых итераций.
    /// Система сначала приводится к виду x = alpha * x + beta.
    /// </summary>
    public static class IterativeSolver
    {
        public static float[] SolveIterative(float[,] A, float[] b, float epsilon, int maxIterations, bool checkConvergence = false)
        {
            int n = A.GetLength(0);
            float[,] alpha = new float[n, n];
            float[] beta = new float[n];

            // Формирование матрицы alpha и вектора beta по формулам:
            // beta[i] = b[i] / A[i,i] и alpha[i,j] = -A[i,j] / A[i,i] (при j != i, alpha[i,i]=0)
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

            // Опциональная проверка условия сходимости (||alpha||∞ < 1)
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

            // Итерационный процесс: начальное приближение x^(0) = beta
            float[] xOld = new float[n];
            float[] xNew = new float[n];
            Array.Copy(beta, xOld, n);

            int iterations = 0;
            while (iterations < maxIterations)
            {
                for (int i = 0; i < n; i++)
                {
                    float sum = 0;
                    for (int j = 0; j < n; j++)
                        sum += alpha[i, j] * xOld[j];
                    xNew[i] = beta[i] + sum;
                }

                float diff = 0;
                for (int i = 0; i < n; i++)
                    diff = Math.Max(diff, Math.Abs(xNew[i] - xOld[i]));

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
    /// Класс для решения трёхдиагональной системы методом прогонки (метод Томаса).
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
            Console.WriteLine("1 - Метод Гаусса (с выбором/без выбора главного элемента)");
            Console.WriteLine("2 - Итерационный метод (метод простых итераций)");
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

                Console.WriteLine($"Введите расширенную матрицу [A|b] построчно (каждая строка содержит {n + 1} чисел, разделенных пробелами):");
                for (int i = 0; i < n; i++)
                {
                    Console.Write($"Строка {i + 1}: ");
                    string[] tokens = Console.ReadLine().Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
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
                Console.WriteLine("2 - С выбором главного элемента");
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

                Console.WriteLine("\nВведите коэффициенты матрицы A (по строкам, разделенные пробелами):");
                for (int i = 0; i < n; i++)
                {
                    Console.Write($"Строка {i + 1}: ");
                    string[] tokens = Console.ReadLine().Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (tokens.Length != n)
                    {
                        Console.WriteLine("Неверное количество чисел. Попробуйте снова.");
                        i--;
                        continue;
                    }
                    for (int j = 0; j < n; j++)
                        A[i, j] = float.Parse(tokens[j], CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВведите вектор свободных членов b (n чисел, разделенных пробелами):");
                Console.Write("b: ");
                {
                    string[] tokens = Console.ReadLine().Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (tokens.Length != n)
                        throw new Exception("Неверное количество чисел во векторе b.");
                    for (int i = 0; i < n; i++)
                        b[i] = float.Parse(tokens[i], CultureInfo.InvariantCulture);
                }

                Console.Write("\nВведите требуемую точность (например, 0.001): ");
                float epsilon = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                Console.Write("Введите максимальное число итераций: ");
                int maxIterations = int.Parse(Console.ReadLine());
                Console.Write("Проверять условие сходимости (||alpha|| < 1)? (1 - Да, 0 - Нет): ");
                bool checkConv = int.Parse(Console.ReadLine()) == 1;

                float[] solution = IterativeSolver.SolveIterative(A, b, epsilon, maxIterations, checkConv);
                Console.WriteLine("\nРешение системы методом итераций:");
                for (int i = 0; i < solution.Length; i++)
                    Console.WriteLine($"x[{i + 1}] = {solution[i]}");
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

                Console.WriteLine("\nВведите элементы главной диагонали d[i] (i = 0 .. n-1):");
                for (int i = 0; i < n; i++)
                {
                    Console.Write($"d[{i}] = ");
                    d[i] = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВведите элементы поддиагонали c[i] (i = 1 .. n-1, c[0] не используется):");
                for (int i = 1; i < n; i++)
                {
                    Console.Write($"c[{i}] = ");
                    c[i] = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВведите элементы наддиагонали e[i] (i = 0 .. n-2, e[n-1] не используется):");
                for (int i = 0; i < n - 1; i++)
                {
                    Console.Write($"e[{i}] = ");
                    e[i] = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }

                Console.WriteLine("\nВведите вектор правых частей b[i] (i = 0 .. n-1):");
                for (int i = 0; i < n; i++)
                {
                    Console.Write($"b[{i}] = ");
                    b[i] = float.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }

                float[] solution = TridiagonalSolver.Solve(c, d, e, b);
                Console.WriteLine("\nРешение трёхдиагональной системы:");
                for (int i = 0; i < solution.Length; i++)
                    Console.WriteLine($"x[{i}] = {solution[i]}");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Ошибка: " + ex.Message);
            }
        }
    }
}
