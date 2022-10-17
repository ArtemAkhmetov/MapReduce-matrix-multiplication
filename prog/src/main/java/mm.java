import java.io.*;
//import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.net.URI;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class mm {

    public static class mmMapper1
            extends Mapper<LongWritable, Text, Text, Text> {
        private int g;
        private int A_rows_g;
        private int A_cols_g;
        private int B_cols_g;
        private String tags;

        @Override
        protected void setup(Context context) {
            int A_rows, A_cols, B_cols;
            Configuration conf = context.getConfiguration();
            A_rows = Integer.parseInt(conf.get("mm.A_rows"));
            A_cols = Integer.parseInt(conf.get("mm.A_cols"));
            B_cols = Integer.parseInt(conf.get("mm.B_cols"));
            g = conf.getInt("mm.groups", 1);
            A_rows_g = A_rows / g;
            A_cols_g = A_cols / g;
            B_cols_g = B_cols / g;
            tags = conf.get("mm.tags","ABC");
        }

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] content = value.toString().split("\t");
            float elem = Float.parseFloat(content[3]);
            int I, J, K;
            int iInd, jInd, kInd;
            String outputKey, outputValue;
            if (tags.startsWith(content[0])) {
                iInd = Integer.parseInt(content[1]);
                jInd = Integer.parseInt(content[2]);
                I = Math.min(iInd / A_rows_g, g - 1);
                J = Math.min(jInd / A_cols_g, g - 1);
                for (int k = 0; k < g; ++k) {
                    outputKey = I + "\t" + J + "\t" + k;
                    outputValue = content[0] + "\t" + iInd + "\t" + jInd + "\t" + elem;
                    context.write(new Text(outputKey), new Text(outputValue));
                }
            }
            else {
                jInd = Integer.parseInt(content[1]);
                kInd = Integer.parseInt(content[2]);
                J = Math.min(jInd / A_cols_g, g - 1);
                K = Math.min(kInd / B_cols_g, g - 1);
                for (int i = 0; i < g; ++i) {
                    outputKey = i + "\t" + J + "\t" + K;
                    outputValue = content[0] + "\t"+ jInd + "\t"+ kInd + "\t"+ elem;
                    context.write(new Text(outputKey), new Text(outputValue));
                }
            }
        }
    }

    public static class mmReducer1
            extends Reducer<Text, Text, Text, Text> {
        private int g;
        private int A_rows_g;
        private int A_cols_g;
        private int B_cols_g;
        private String tags;
        private int A_rows_g_mod;
        private int A_cols_g_mod;
        private int B_cols_g_mod;
        private final Text outputValue = new Text();

        @Override
        protected void setup(Context context) {
            int A_rows, A_cols, B_cols;
            Configuration conf = context.getConfiguration();
            g = conf.getInt("mm.groups", 1);
            tags = conf.get("mm.tags","ABC");
            A_rows = Integer.parseInt(conf.get("mm.A_rows"));
            A_cols = Integer.parseInt(conf.get("mm.A_cols"));
            B_cols = Integer.parseInt(conf.get("mm.B_cols"));
            A_rows_g = A_rows / g;
            A_cols_g = A_cols / g;
            B_cols_g = B_cols / g;
            A_rows_g_mod = A_rows % g;
            A_cols_g_mod = A_cols % g;
            B_cols_g_mod = B_cols % g;
        }

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            HashMap<String, Float> A = new HashMap<>();
            HashMap<String, Float> B = new HashMap<>();
            String mapKey, outputKey, A_key, B_key;
            String[] content;
            String[] inputKey = key.toString().split("\t");
            float elem, sum;
            int I, J, K, ind_i, ind_j, ind_k;
            int idI = Integer.parseInt(inputKey[0]);
            int idJ = Integer.parseInt(inputKey[1]);
            int idK = Integer.parseInt(inputKey[2]);
            for (Text val : values) {
                content = val.toString().split("\t");
                elem = Float.parseFloat(content[3]);
                mapKey = content[1] + "\t" + content[2];
                if (tags.startsWith(content[0])) {
                    A.put(mapKey, elem);
                }
                else {
                    B.put(mapKey, elem);
                }
            }

            I = (idI == g-1) ? A_rows_g + A_rows_g_mod : A_rows_g;
            J = (idJ == g-1) ? A_cols_g + A_cols_g_mod : A_cols_g;
            K = (idK == g-1) ? B_cols_g + B_cols_g_mod : B_cols_g;

            for (int i = 0; i < I; ++i) {
                ind_i = i + A_rows_g * idI;
                for (int k = 0; k < K; ++k) {
                    ind_k = k + B_cols_g * idK;
                    sum = 0.0f;
                    for (int j = 0; j < J; ++j) {
                        ind_j = j + A_cols_g * idJ;
                        A_key = ind_i + "\t" + ind_j;
                        B_key = ind_j + "\t" + ind_k;
                        if (A.containsKey(A_key) && B.containsKey(B_key)) {
                            sum += A.get(A_key) * B.get(B_key);
                        }
                    }
                    outputKey = ind_i + "\t" + ind_k;
                    if (sum != 0.0f) {
                        outputValue.set(String.valueOf(sum));
                        context.write(new Text(outputKey), outputValue);
                    }
                }
            }
        }
    }

    public static class mmMapper2
            extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] content = value.toString().split("\t");
            String outputKey  = content[0] + "\t" + content[1];
            String outputValue = content[2];
            context.write(new Text(outputKey), new Text(outputValue));
        }
    }

    public static class mmReducer2
            extends Reducer<Text, Text, Text, Text> {

        private final Text outputValue = new Text();
        private String floatFormat;
        private char[] tags;

        @Override
        protected void setup(Context context){
            Configuration conf = context.getConfiguration();
            floatFormat = conf.get("mm.float-format", "%.3f");
            tags = conf.get("mm.tags","ABC").toCharArray();
        }

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            float sum = 0.0f;
            String outputKey;
            outputKey = tags[2] + "\t" + key;
            for (Text val : values) {
                sum += Float.parseFloat(val.toString());
            }
            outputValue.set(String.format(Locale.US, floatFormat, sum));
            context.write(new Text(outputKey), outputValue);
        }
    }

    public static void main(String[] args) throws Exception {
        //long start = new Date().getTime();
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        GenericOptionsParser ops = new GenericOptionsParser(conf, args);
        conf = ops.getConfiguration();
        int num_reducers = conf.getInt("mapred.reduce.tasks", 1);
        if (otherArgs.length != 3) {
            System.err.println("Usage: hadoop jar mm.jar mm -conf config.xml <path to matrix A> <path to matrix B> <path to matrix C>");
            System.exit(2);
        }
        FileSystem fs = FileSystem.get(URI.create(otherArgs[0]), conf);
        Path path = new Path(otherArgs[0] + "/size");
        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));
        String content = "";
        try {
            content = reader.readLine();
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            reader.close();
        }
        String[] size = content.split("\t");
        int rows = Integer.parseInt(size[0]), cols = Integer.parseInt(size[1]);
        if (Math.min(rows, cols) < conf.getInt("mm.groups", 1)) {
            System.err.println("The number of rows and columns must be not less than the number of groups");
            System.exit(2);
        }
        conf.setInt("mm.A_rows", rows);
        conf.setInt("mm.A_cols", cols);
        conf.setInt("mm.C_rows", rows);
        conf.setInt("mm.B_rows", cols);
        fs = FileSystem.get(URI.create(otherArgs[1]), conf);
        path = new Path(otherArgs[1] + "/size");
        reader = new BufferedReader(new InputStreamReader(fs.open(path)));
        content = "";
        try {
            content = reader.readLine();
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            reader.close();
        }
        size = content.split("\t");
        cols = Integer.parseInt(size[1]);
        if (cols < conf.getInt("mm.groups", 1)) {
            System.err.println("The number of rows and columns must be not less than the number of groups");
            System.exit(2);
        }
        conf.setInt("mm.B_cols", cols);
        conf.setInt("mm.C_cols", cols);
        fs = FileSystem.get(URI.create(otherArgs[2]), conf);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(otherArgs[2] + "/size"))));
        String input = conf.get("mm.C_rows") + "\t" + conf.get("mm.C_cols") + "\n";
        writer.write(input);
        writer.close();
        Job job1 = Job.getInstance(conf);
        job1.setJobName("mm");
        job1.setJarByClass(mm.class);
        job1.setMapperClass(mmMapper1.class);
        job1.setReducerClass(mmReducer1.class);
        job1.setNumReduceTasks(num_reducers);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        job1.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.addInputPath(job1, new Path(otherArgs[0] + "/data"));
        FileInputFormat.addInputPath(job1, new Path(otherArgs[1] + "/data"));
        FileOutputFormat.setOutputPath(job1, new Path(otherArgs[1] + "/temp"));
        if (!job1.waitForCompletion(true))
        {
            System.exit(1);
        }
        Job job2 = Job.getInstance(conf);
        job2.setJobName("mm");
        job2.setJarByClass(mm.class);
        job2.setMapperClass(mmMapper2.class);
        job2.setReducerClass(mmReducer2.class);
        job2.setNumReduceTasks(num_reducers);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        job2.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.addInputPath(job2, new Path(otherArgs[1] + "/temp"));
        FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2] + "/data"));
        System.exit(job2.waitForCompletion(true) ? 0 : 1);

        /*boolean status = job2.waitForCompletion(true);
        long end = new Date().getTime();
        System.out.println("Job took "+(end-start) + "milliseconds");
        System.exit(0);*/
    }
}