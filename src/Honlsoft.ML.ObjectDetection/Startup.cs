using Honlsoft.ML.ObjectDetection.Models.Yolov4;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.OpenApi.Models;

namespace Honlsoft.ML.ObjectDetection
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {

            services.AddControllers();
            services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new OpenApiInfo { Title = "Honlsoft.ML.ObjectDetection", Version = "v1" });
            });

            
            // Setup the Yolov4 Model.
            services.AddSingleton<IObjectDetector, YoloObjectDetector>();
            services.AddSingleton<YoloPostProcessor>();
            services.AddSingleton<YoloPredictionEngineFactory>();
            services.Configure<YoloOptions>(Configuration.GetSection("Yolo"));
            
            // Setup the dependencies for the controller
            services.Configure<SourceOptions>(Configuration.GetSection("Source"));
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
                app.UseSwagger();
                app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "Honlsoft.ML.ObjectDetection v1"));
            }

            //app.UseHttpsRedirection();

            app.UseRouting();

            app.UseAuthorization();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
            });
        }
    }
}
