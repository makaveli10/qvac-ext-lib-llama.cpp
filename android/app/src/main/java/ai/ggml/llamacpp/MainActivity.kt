package ai.ggml.llamacpp


import ai.ggml.llamacpp.ui.theme.LlamaAndroidTheme
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.DownloadManager
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.text.format.Formatter
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.getSystemService
import androidx.core.net.toUri
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import java.io.File

class MainActivity(
    activityManager: ActivityManager? = null,
    downloadManager: DownloadManager? = null,
    clipboardManager: ClipboardManager? = null,
): ComponentActivity() {
    private val tag: String? = this::class.simpleName

    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }

    private val viewModel: MainViewModel by viewModels()

    // Get a MemoryInfo object for the device's current memory status.
    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)

        viewModel.log("Current memory: $free / $total")
        viewModel.log("Downloads directory: ${getExternalFilesDir(null)}")

        val extFilesDir = getExternalFilesDir(null)

        val models = listOf(
            Downloadable(
                "Phi-2 7B (Q4_0, 1.6 GiB)",
                "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true".toUri(),
                File(extFilesDir, "phi-2-q4_0.gguf"),
            ),
            Downloadable(
                "TinyLlama 1.1B (f16, 2.2 GiB)",
                "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true".toUri(),
                File(extFilesDir, "tinyllama-1.1-f16.gguf"),
            ),
            Downloadable(
                "Phi 2 DPO (Q3_K_M, 1.48 GiB)",
                "https://huggingface.co/TheBloke/phi-2-dpo-GGUF/resolve/main/phi-2-dpo.Q3_K_M.gguf?download=true".toUri(),
                File(extFilesDir, "phi-2-dpo.Q3_K_M.gguf")
            ),
        )

        val actionCopy: () -> Unit = {
            viewModel.messages.joinToString("\n").let {
                clipboardManager.setPrimaryClip(ClipData.newPlainText("", it))
            }
        }

        setContent {
            LlamaAndroidTheme {
                // A surface container using the 'background' color from the theme

                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainCompose(
                        viewModel,
                        actionCopy,
                        downloadManager,
                        models,
                    )
                }

            }
        }
    }
}
@Composable
fun ChatScreen(
    viewModel: MainViewModel,
    actionCopy: () -> Unit
) {
    Column {
        val scrollState = rememberLazyListState()

        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            LazyColumn(state = scrollState) {
                items(viewModel.messages) {
                    Text(
                        it,
                        style = MaterialTheme.typography.bodyLarge.copy(color = LocalContentColor.current),
                        modifier = Modifier.padding(16.dp)
                    )
                }
            }
        }
        Row (
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            OutlinedTextField(
                value = viewModel.message,
                onValueChange = { viewModel.updateMessage(it) },
                label = { Text("Prompt") },
                modifier = Modifier.weight(1f),
                singleLine = true,
            )
            Button(
                onClick = { viewModel.send() },
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Send",
                )
            }
        }
        Row {
            Button({ viewModel.bench(8, 4, 1) }) { Text("Bench") }
            Button({ viewModel.clear() }) { Text("Clear") }
            Button(actionCopy) { Text("Copy") }
        }


    }
}

enum class Destination(
    val route: String,
    val label: String,
    val icon: ImageVector,
    val contentDescription: String
) {
    CHAT("chat", "Chat", Icons.AutoMirrored.Filled.Send, "Chat"),
    BENCHMARK("benchmark", "Benchmark", Icons.AutoMirrored.Filled.Send, "Benchmark"),
    MODELS("models", "Models", Icons.AutoMirrored.Filled.Send, "Models")
}

@Composable
fun BenchmarkScreen(modifier: Modifier = Modifier) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text("Benchmark Screen")
    }
}

@Composable
fun ModelsScreen(
    viewModel: MainViewModel,
    dm: DownloadManager?,
    models: List<Downloadable>
) {
    Column (
        modifier = Modifier.fillMaxSize()
    ) {
        for (model in models) {
            Downloadable.Button(viewModel, dm, model)
        }
    }
}

@Composable
fun AppNavHost(
    navController: NavHostController,
    startDestination: Destination,
    viewModel: MainViewModel,
    actionCopy: () -> Unit,
    dm: DownloadManager?,
    models: List<Downloadable>
) {
    NavHost(
        navController,
        startDestination = startDestination.route
    ) {
        Destination.entries.forEach { destination ->
            composable(destination.route) {
                when (destination) {
                    Destination.CHAT -> ChatScreen(viewModel, actionCopy)
                    Destination.BENCHMARK -> BenchmarkScreen()
                    Destination.MODELS -> ModelsScreen(viewModel, dm, models)
                }
            }
        }
    }
}

@Composable
fun MainCompose(
    viewModel: MainViewModel,
    actionCopy: () -> Unit,
    dm: DownloadManager?,
    models: List<Downloadable>
) {
    val navController = rememberNavController()
    val startDestination = Destination.CHAT
    var selectedDestination by rememberSaveable { mutableIntStateOf(startDestination.ordinal) }

    Scaffold() { contentPadding ->
        Column() {
            PrimaryTabRow(selectedTabIndex = selectedDestination, modifier = Modifier.padding(contentPadding)) {
                Destination.entries.forEachIndexed { index, destination ->
                    Tab(
                        selected = selectedDestination == index,
                        onClick = {
                            navController.navigate(route = destination.route)
                            selectedDestination = index
                        },
                        text = {
                            Text(
                                text = destination.label,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    )
                }
            }
            AppNavHost(navController, startDestination, viewModel, actionCopy, dm, models)
        }
    }
}

@SuppressLint("ViewModelConstructorInComposable")
@Preview
@Composable
fun MainComposePreview() {
    val viewModel = MainViewModel()
    val actionCopy: () -> Unit = {}

    val models = listOf(
        Downloadable(
            "Phi-2 7B (Q4_0, 1.6 GiB)",
            "".toUri(),
            File(""),
        ),
        Downloadable(
            "TinyLlama 1.1B (f16, 2.2 GiB)",
            "".toUri(),
            File(""),
        ),
        Downloadable(
            "Phi 2 DPO (Q3_K_M, 1.48 GiB)",
            "".toUri(),
            File("")
        ),
    )
    MainCompose(
        viewModel,
        actionCopy,
        null,
        models
    )
}
